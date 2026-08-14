[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ReleaseDirectory,
    [string]$PythonPath = "",
    [string]$ExpectedVersion = "AFC2.0.2alpha",
    [string]$ProjectRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-NormalizedPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$BasePath = ""
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    if (-not $BasePath) {
        $BasePath = (Get-Location).Path
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Parent,
        [switch]$AllowEqual
    )

    $candidatePath = (Get-NormalizedPath -Path $Candidate).TrimEnd([char[]]"\/")
    $parentPath = (Get-NormalizedPath -Path $Parent).TrimEnd([char[]]"\/")
    if ($AllowEqual -and $candidatePath.Equals($parentPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $parentPath + [System.IO.Path]::DirectorySeparatorChar
    return $candidatePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-FileSha256WithRetry {
    param([Parameter(Mandatory = $true)][string]$Path)

    $lastFailure = $null
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        catch {
            $lastFailure = $_
            if ($attempt -lt 10) {
                Start-Sleep -Milliseconds 500
            }
        }
    }
    throw "多次重试后仍无法读取文件 SHA-256: $Path；$($lastFailure.Exception.Message)"
}

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = (Get-NormalizedPath -Path $BasePath).TrimEnd([char[]]"\/") + [System.IO.Path]::DirectorySeparatorChar
    $targetFull = Get-NormalizedPath -Path $TargetPath
    $relativeUri = ([System.Uri]$baseFull).MakeRelativeUri([System.Uri]$targetFull)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace("\", "/")
}

function Resolve-VerificationPython {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $resolvedRequest = Get-NormalizedPath -Path $RequestedPath
        if (-not (Test-Path -LiteralPath $resolvedRequest -PathType Leaf)) {
            throw "指定的 Python 不存在: $resolvedRequest"
        }
        return (Resolve-Path -LiteralPath $resolvedRequest).Path
    }

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:CONDA_PREFIX -and (Split-Path -Leaf $env:CONDA_PREFIX) -ieq "AFC") {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add($pythonCommand.Source)
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "未找到验证所需的 Python；请通过 -PythonPath 指定 AFC 环境解释器。"
}

if (-not $ProjectRoot) {
    $ProjectRoot = Join-Path $PSScriptRoot "..\.."
}
$ProjectRoot = Get-NormalizedPath -Path $ProjectRoot
$ReleaseDirectory = Get-NormalizedPath -Path $ReleaseDirectory
if (-not (Test-Path -LiteralPath $ReleaseDirectory -PathType Container)) {
    throw "发布目录不存在: $ReleaseDirectory"
}

$ResolvedPython = Resolve-VerificationPython -RequestedPath $PythonPath
$PyInstallerVersion = (& $ResolvedPython -c "import PyInstaller; print(PyInstaller.__version__)") | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "验证环境无法导入 PyInstaller。"
}
$PyInstallerVersion = $PyInstallerVersion.Trim()
if ($PyInstallerVersion -ne "6.12.0") {
    throw "验证要求 PyInstaller 6.12.0，当前为 $PyInstallerVersion。"
}

$ExecutablePath = Join-Path $ReleaseDirectory "$ExpectedVersion.exe"
$InternalDirectory = Join-Path $ReleaseDirectory "_internal"
$ReleaseNotesPath = Join-Path $ReleaseDirectory "发布说明.txt"
$ManifestPath = Join-Path $ReleaseDirectory "build-manifest.json"
$ChecksumsPath = Join-Path $ReleaseDirectory "SHA256SUMS.txt"

foreach ($requiredFile in @($ExecutablePath, $ReleaseNotesPath, $ManifestPath, $ChecksumsPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "发布目录缺少必需文件: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath $InternalDirectory -PathType Container)) {
    throw "发布目录缺少 PyInstaller _internal 目录。"
}

$RequiredMklRuntimeFiles = @(
    "mkl_rt.2.dll",
    "mkl_core.2.dll",
    "mkl_intel_thread.2.dll",
    "mkl_def.2.dll",
    "mkl_vml_def.2.dll",
    "libiomp5md.dll"
)
foreach ($runtimeName in $RequiredMklRuntimeFiles) {
    $runtimePath = Join-Path $InternalDirectory $runtimeName
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        throw "发布目录缺少 NumPy/Pandas 所需 MKL 运行库: $runtimeName"
    }
}

$ReleaseNotes = [System.IO.File]::ReadAllText($ReleaseNotesPath, [System.Text.Encoding]::UTF8)
if ($ReleaseNotes -match "\{\{[A-Z_]+\}\}") {
    throw "发布说明仍包含未替换的模板占位符。"
}

$Manifest = [System.IO.File]::ReadAllText($ManifestPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
if ($Manifest.product -cne $ExpectedVersion) {
    throw "构建清单版本不匹配: $($Manifest.product)"
}
if ([string]$Manifest.git_commit -notmatch "^[0-9a-fA-F]{40}$") {
    throw "构建清单缺少有效的 40 位 Git 提交。"
}
if ([string]$Manifest.dependencies.PyInstaller -ne "6.12.0") {
    throw "构建清单中的 PyInstaller 版本不是 6.12.0。"
}

$ForbiddenRootSegments = @("config", "data", "profiles", "output", "src", "tests", "archive")
$ForbiddenDependencyTokens = @("sklearn", "scikit_learn", "scipy", "joblib", "threadpoolctl")
$RuntimeInputNames = @(
    "SampleData.csv",
    "SampleData.txt",
    "iipinc.txt",
    "SampleData.rg",
    "ProcessDataPath.txt"
)
$RuntimeInputPatterns = @("*.csv", "SampleData*.txt")

function Test-IsRootRuntimeInput {
    param(
        [Parameter(Mandatory = $true)]$Entry,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    if ($Entry.PSIsContainer -or $RelativePath -cne $Entry.Name) {
        return $false
    }
    if ($RuntimeInputNames -icontains $Entry.Name) {
        return $true
    }
    foreach ($pattern in $RuntimeInputPatterns) {
        if ($Entry.Name -like $pattern) {
            return $true
        }
    }
    return $false
}

$ForbiddenFiles = New-Object System.Collections.Generic.List[string]
$AllEntries = Get-ChildItem -LiteralPath $ReleaseDirectory -Recurse -Force
foreach ($entry in $AllEntries) {
    $relativePath = Get-PortableRelativePath -BasePath $ReleaseDirectory -TargetPath $entry.FullName
    $isRootRuntimeInput = Test-IsRootRuntimeInput -Entry $entry -RelativePath $relativePath
    if ($isRootRuntimeInput) {
        continue
    }
    $segments = @($relativePath -split "[/\\]")
    foreach ($segment in $segments) {
        if ($ForbiddenRootSegments -icontains $segment) {
            $ForbiddenFiles.Add("禁止目录段 '$segment': $relativePath")
            break
        }
    }

    foreach ($token in $ForbiddenDependencyTokens) {
        if ($relativePath.IndexOf($token, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            $ForbiddenFiles.Add("禁止依赖 '$token': $relativePath")
            break
        }
    }

    if (-not $entry.PSIsContainer) {
        $lowerName = $entry.Name.ToLowerInvariant()
        if ($lowerName.EndsWith(".py") -or $lowerName.EndsWith(".kcke") -or $lowerName.EndsWith(".kcke.json")) {
            $ForbiddenFiles.Add("禁止文件类型: $relativePath")
        }
        if ($lowerName -in @("app_config.json", "ideal_store.json")) {
            $ForbiddenFiles.Add("禁止项目/用户数据: $relativePath")
        }
        if ($lowerName.StartsWith("processinfo") -or $lowerName.EndsWith(".log")) {
            $ForbiddenFiles.Add("禁止样例或日志: $relativePath")
        }
    }
}
if ($ForbiddenFiles.Count -gt 0) {
    throw "发布目录 denylist 检查失败:`n$($ForbiddenFiles -join [Environment]::NewLine)"
}

$ChecksumRecords = @{}
$ChecksumLines = [System.IO.File]::ReadAllLines($ChecksumsPath, [System.Text.Encoding]::UTF8)
foreach ($line in $ChecksumLines) {
    if (-not $line.Trim()) {
        continue
    }
    if ($line -notmatch "^([0-9a-fA-F]{64})  (.+)$") {
        throw "SHA256SUMS.txt 存在无效行: $line"
    }
    $expectedHash = $Matches[1].ToLowerInvariant()
    $relativePath = $Matches[2]
    if ([System.IO.Path]::IsPathRooted($relativePath) -or @($relativePath -split "[/\\]") -contains "..") {
        throw "校验清单包含不安全路径: $relativePath"
    }
    if ($ChecksumRecords.ContainsKey($relativePath)) {
        throw "校验清单包含重复路径: $relativePath"
    }

    $candidatePath = Get-NormalizedPath -Path ($relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar)) -BasePath $ReleaseDirectory
    if (-not (Test-PathWithin -Candidate $candidatePath -Parent $ReleaseDirectory)) {
        throw "校验路径越出发布目录: $relativePath"
    }
    if (-not (Test-Path -LiteralPath $candidatePath -PathType Leaf)) {
        throw "校验清单中的文件不存在: $relativePath"
    }
    $actualHash = Get-FileSha256WithRetry -Path $candidatePath
    if ($actualHash -cne $expectedHash) {
        throw "SHA-256 不匹配: $relativePath"
    }
    $ChecksumRecords[$relativePath] = $true
}

$UnlistedFiles = New-Object System.Collections.Generic.List[string]
$ReleaseFiles = Get-ChildItem -LiteralPath $ReleaseDirectory -Recurse -Force -File |
    Where-Object {
        if ($_.FullName -eq $ChecksumsPath) {
            return $false
        }
        $relativePath = Get-PortableRelativePath -BasePath $ReleaseDirectory -TargetPath $_.FullName
        return -not (Test-IsRootRuntimeInput -Entry $_ -RelativePath $relativePath)
    }
foreach ($file in $ReleaseFiles) {
    $relativePath = Get-PortableRelativePath -BasePath $ReleaseDirectory -TargetPath $file.FullName
    if (-not $ChecksumRecords.ContainsKey($relativePath)) {
        $UnlistedFiles.Add($relativePath)
    }
}
if ($UnlistedFiles.Count -gt 0) {
    throw "存在未纳入 SHA256SUMS.txt 的文件:`n$($UnlistedFiles -join [Environment]::NewLine)"
}

$ArchiveViewerCommand = "from PyInstaller.utils.cliutils.archive_viewer import run; run()"
$ArchiveListing = (& $ResolvedPython -c $ArchiveViewerCommand -l -r -b --log-level ERROR $ExecutablePath 2>&1) | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 PyInstaller CArchive/PYZ。"
}
$ForbiddenArchiveModules = @(
    "project.app",
    "project.pit_model",
    "sklearn",
    "scipy",
    "joblib",
    "threadpoolctl",
    "pkg_resources"
)
$ArchiveModules = @(
    $ArchiveListing -split "`r?`n" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
foreach ($moduleName in $ForbiddenArchiveModules) {
    $matchedModule = $ArchiveModules | Where-Object {
        $_.Equals($moduleName, [System.StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith($moduleName + ".", [System.StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if ($matchedModule) {
        throw "PyInstaller 内部归档包含禁止模块: $matchedModule"
    }
}
$RequiredArchiveModules = @(
    "project.prediction_support",
    "project.release_prediction"
)
foreach ($moduleName in $RequiredArchiveModules) {
    $matchedModule = $ArchiveModules | Where-Object {
        $_.Equals($moduleName, [System.StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if (-not $matchedModule) {
        throw "PyInstaller 内部归档缺少发布预测模块: $moduleName"
    }
}

$ZipInspection = @'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
forbidden = ("project/app", "project/pit_model", "sklearn", "scipy", "joblib", "threadpoolctl")
violations = []
for archive in root.rglob("*.zip"):
    try:
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                normalized = name.replace("\\", "/").lower()
                module_path = normalized.rsplit(".", 1)[0]
                forbidden_module = any(
                    module_path == prefix or module_path.startswith(prefix + "/")
                    for prefix in forbidden
                )
                if normalized.endswith(".py") or forbidden_module:
                    violations.append(f"{archive.relative_to(root)}::{name}")
    except zipfile.BadZipFile:
        violations.append(f"无效ZIP::{archive.relative_to(root)}")
if violations:
    print("\n".join(violations[:100]))
    raise SystemExit(1)
'@
$ZipInspectionOutput = (& $ResolvedPython -c $ZipInspection $ReleaseDirectory 2>&1) | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "发布目录 ZIP 内容检查失败:`n$ZipInspectionOutput"
}

$PythonPrefix = (& $ResolvedPython -c "import sys; print(sys.prefix)") | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "无法读取验证 Python 前缀。"
}
$DeveloperPaths = @(
    $ProjectRoot,
    [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile),
    $PythonPrefix.Trim()
) | Where-Object { $_ -and $_.Length -ge 4 } | Select-Object -Unique

$AbsolutePathScanner = @'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
needles = [value for value in sys.argv[2:] if value]
patterns = []
for needle in needles:
    normalized = needle.rstrip("\\/")
    for value in {normalized, normalized.replace("\\", "/")}:
        patterns.append((needle, value.encode("utf-8", "ignore")))
        patterns.append((needle, value.encode("utf-16-le", "ignore")))
patterns = [(label, pattern) for label, pattern in patterns if pattern]
max_pattern = max((len(pattern) for _, pattern in patterns), default=1)
hits = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    tail = b""
    matched = False
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                data = tail + chunk
                for label, pattern in patterns:
                    if pattern in data:
                        hits.append(f"{path.relative_to(root)} -> {label}")
                        matched = True
                        break
                if matched:
                    break
                tail = data[-(max_pattern - 1):] if max_pattern > 1 else b""
    except OSError as exc:
        hits.append(f"无法扫描 {path.relative_to(root)}: {exc}")
if hits:
    print("\n".join(hits[:100]))
    raise SystemExit(1)
'@
$PathScanOutput = (& $ResolvedPython -c $AbsolutePathScanner $ReleaseDirectory @DeveloperPaths 2>&1) | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "发布目录包含开发机绝对路径或存在无法扫描的文件:`n$PathScanOutput"
}

# windowed EXE 的原生 DLL 加载失败不会产生 Python traceback。复制到临时目录后
# 执行隐藏启动自检，确保 NumPy/Pandas、Tk 和发布应用都能完成初始化。
$SmokeParent = Get-NormalizedPath -Path ([System.IO.Path]::GetTempPath())
$SmokeRoot = Join-Path $SmokeParent ("AFC2-release-smoke-" + [guid]::NewGuid().ToString("N"))
$SmokeReleaseDirectory = Join-Path $SmokeRoot $ExpectedVersion
$PreviousSmokeFlag = $env:AFC_RELEASE_SMOKE_TEST
$PreviousSuppressMessageboxes = $env:SUPPRESS_MESSAGEBOXES
try {
    New-Item -ItemType Directory -Path $SmokeReleaseDirectory -Force | Out-Null
    Get-ChildItem -LiteralPath $ReleaseDirectory -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $SmokeReleaseDirectory -Recurse -Force
    }
    $SmokeExecutable = Join-Path $SmokeReleaseDirectory "$ExpectedVersion.exe"
    $env:AFC_RELEASE_SMOKE_TEST = "1"
    $env:SUPPRESS_MESSAGEBOXES = "1"
    $SmokeProcess = Start-Process `
        -FilePath $SmokeExecutable `
        -WorkingDirectory $SmokeReleaseDirectory `
        -WindowStyle Hidden `
        -PassThru
    if (-not $SmokeProcess.WaitForExit(30000)) {
        Stop-Process -Id $SmokeProcess.Id -Force -ErrorAction SilentlyContinue
        throw "发布 EXE 隐藏启动自检超时。"
    }
    if ($SmokeProcess.ExitCode -ne 0) {
        $SmokeLogPath = Join-Path $SmokeReleaseDirectory "startup-error.log"
        $SmokeLogText = ""
        if (Test-Path -LiteralPath $SmokeLogPath -PathType Leaf) {
            $SmokeLogText = [System.IO.File]::ReadAllText($SmokeLogPath, [System.Text.Encoding]::UTF8)
        }
        throw "发布 EXE 隐藏启动自检失败，退出码 $($SmokeProcess.ExitCode)。`n$SmokeLogText"
    }
    $UnexpectedOutputDirectory = Join-Path $SmokeReleaseDirectory "output"
    if (Test-Path -LiteralPath $UnexpectedOutputDirectory) {
        throw "发布 EXE 启动后仍创建了 output 子目录: $UnexpectedOutputDirectory"
    }
}
finally {
    if ($null -eq $PreviousSmokeFlag) {
        Remove-Item Env:AFC_RELEASE_SMOKE_TEST -ErrorAction SilentlyContinue
    }
    else {
        $env:AFC_RELEASE_SMOKE_TEST = $PreviousSmokeFlag
    }
    if ($null -eq $PreviousSuppressMessageboxes) {
        Remove-Item Env:SUPPRESS_MESSAGEBOXES -ErrorAction SilentlyContinue
    }
    else {
        $env:SUPPRESS_MESSAGEBOXES = $PreviousSuppressMessageboxes
    }
    if (Test-Path -LiteralPath $SmokeRoot) {
        $ResolvedSmokeRoot = (Resolve-Path -LiteralPath $SmokeRoot).Path
        if (-not (Test-PathWithin -Candidate $ResolvedSmokeRoot -Parent $SmokeParent)) {
            throw "临时启动自检目录越界，拒绝清理: $ResolvedSmokeRoot"
        }
        $SmokeItem = Get-Item -LiteralPath $ResolvedSmokeRoot -Force
        if (($SmokeItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "临时启动自检目录是重解析点，拒绝清理: $ResolvedSmokeRoot"
        }
        Remove-Item -LiteralPath $ResolvedSmokeRoot -Recurse -Force
    }
}

Write-Host "发布目录验证通过: $ReleaseDirectory"
Write-Host "- 必需目录与文件完整"
Write-Host "- SHA-256 全量校验通过"
Write-Host "- 文件树、ZIP、CArchive/PYZ denylist 通过"
Write-Host "- 未发现当前开发机绝对路径"
Write-Host "- 发布 EXE 隐藏启动自检通过"
