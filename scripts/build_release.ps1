[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$OutputRoot = "",
    [string]$ArchiveDestination = "",
    [switch]$CleanBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RequiredPyInstallerVersion = "6.12.0"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

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

function Remove-SafeDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$AllowedParent
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $resolvedParent = (Resolve-Path -LiteralPath $AllowedParent).Path
    if (-not (Test-PathWithin -Candidate $resolvedPath -Parent $resolvedParent)) {
        throw "拒绝清理允许目录以外的路径: $resolvedPath"
    }
    $targetItem = Get-Item -LiteralPath $resolvedPath -Force
    if (($targetItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        # 同步存储提供程序会给普通目录附加云占位重解析标记，此时
        # LinkType/Target 均为空；它不是会跳出已验证父目录的文件系统链接。
        # Junction/SymbolicLink 等真实链接仍一律拒绝递归清理。
        $hasLinkType = -not [string]::IsNullOrWhiteSpace([string]$targetItem.LinkType)
        $hasLinkTarget = @($targetItem.Target).Count -gt 0 -and -not [string]::IsNullOrWhiteSpace(
            [string](@($targetItem.Target) -join "")
        )
        if ($hasLinkType -or $hasLinkTarget) {
            throw "拒绝递归清理文件系统链接目录: $resolvedPath"
        }
    }
    $lastRemovalFailure = $null
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            Remove-Item -LiteralPath $resolvedPath -Recurse -Force
            return
        }
        catch {
            $lastRemovalFailure = $_
            if ($attempt -lt 10 -and (Test-Path -LiteralPath $resolvedPath)) {
                Start-Sleep -Milliseconds 500
                continue
            }
            if (-not (Test-Path -LiteralPath $resolvedPath)) {
                return
            }
        }
    }
    throw "多次重试后仍无法清理专用构建目录: $resolvedPath；$($lastRemovalFailure.Exception.Message)"
}

function Remove-SafeFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$AllowedParent
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $resolvedParent = (Resolve-Path -LiteralPath $AllowedParent).Path
    if (-not (Test-PathWithin -Candidate $resolvedPath -Parent $resolvedParent)) {
        throw "拒绝清理允许目录以外的文件: $resolvedPath"
    }
    $targetItem = Get-Item -LiteralPath $resolvedPath -Force
    if ($targetItem.PSIsContainer) {
        throw "目标不是文件，拒绝清理: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Force
}

function Resolve-AfcPython {
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

    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaCommand) {
        try {
            $condaJson = (& $condaCommand.Source info --envs --json 2>$null) | Out-String
            if ($LASTEXITCODE -eq 0 -and $condaJson.Trim()) {
                $condaInfo = $condaJson | ConvertFrom-Json
                foreach ($environmentPath in @($condaInfo.envs)) {
                    if ((Split-Path -Leaf $environmentPath) -ieq "AFC") {
                        $candidates.Add((Join-Path $environmentPath "python.exe"))
                    }
                }
            }
        }
        catch {
            # 后续统一由候选路径验证给出错误。
        }
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
    throw "未找到 AFC 环境的 python.exe；请通过 -PythonPath 显式指定。"
}

function Get-BuildEnvironment {
    param([Parameter(Mandatory = $true)][string]$PythonExecutable)

    $probe = @'
import importlib.metadata as metadata
import json
import pathlib
import platform
import sys

packages = ["PyInstaller", "numpy", "pandas", "matplotlib", "scikit-learn"]
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None

print(json.dumps({
    "python_executable": sys.executable,
    "python_prefix": sys.prefix,
    "environment_name": pathlib.Path(sys.prefix).name,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "packages": versions,
}, ensure_ascii=False))
'@

    $jsonOutput = (& $PythonExecutable -c $probe) | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "无法读取 Python 构建环境。"
    }
    return $jsonOutput | ConvertFrom-Json
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

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    [System.IO.File]::WriteAllText($Path, $Content, $Utf8NoBom)
}

function Get-ReleaseProductName {
    param([Parameter(Mandatory = $true)][string]$VersionSource)

    $sourceText = [System.IO.File]::ReadAllText($VersionSource, [System.Text.Encoding]::UTF8)
    $versionMatch = [regex]::Match(
        $sourceText,
        '(?m)^RELEASE_VERSION\s*=\s*["''](?<version>[^"'']+)["'']\s*$'
    )
    if (-not $versionMatch.Success) {
        throw "未在版本文件中找到 RELEASE_VERSION: $VersionSource"
    }
    $version = $versionMatch.Groups["version"].Value
    if ($version -notmatch '^AFC\d+\.\d+\.\d+(?:[-.]?[A-Za-z][0-9A-Za-z.-]*)?$') {
        throw "发布版本格式无效: $version"
    }
    return $version
}

function Get-DebugDirectoryName {
    param([Parameter(Mandatory = $true)][string]$ProductName)

    $nameMatch = [regex]::Match(
        $ProductName,
        '^(?<base>AFC\d+\.\d+\.\d+)(?<stage>[A-Za-z][0-9A-Za-z.-]*)?$'
    )
    if (-not $nameMatch.Success) {
        throw "无法从发布版本生成调试目录名: $ProductName"
    }
    $stage = $nameMatch.Groups["stage"].Value
    if ($stage) {
        $stage = $stage.Substring(0, 1).ToUpperInvariant() + $stage.Substring(1)
    }
    return "$($nameMatch.Groups['base'].Value)$stage-debug"
}

$ProjectRoot = Get-NormalizedPath -Path (Join-Path $PSScriptRoot "..")
$ReleaseArchiveRoot = Get-NormalizedPath -Path (Join-Path (Split-Path -Parent $ProjectRoot) "历史归档/发布版本")
$ReleaseVersionSource = Join-Path $ProjectRoot "src\interval_analysis\release_metadata.py"
$SpecPath = Join-Path $ProjectRoot "packaging\AFC2_onedir.spec"
$ReleaseNotesTemplate = Join-Path $ProjectRoot "packaging\release_notes.txt"
$VerificationScript = Join-Path $ProjectRoot "scripts\verification\verify_release_build.ps1"
$ReleaseEntryPoint = Join-Path $ProjectRoot "src\interval_analysis\release_main.py"

foreach ($requiredFile in @(
    $ReleaseVersionSource,
    $SpecPath,
    $ReleaseNotesTemplate,
    $VerificationScript,
    $ReleaseEntryPoint
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "缺少发布构建所需文件: $requiredFile"
    }
}
$ProductName = Get-ReleaseProductName -VersionSource $ReleaseVersionSource
$DebugDirectoryName = Get-DebugDirectoryName -ProductName $ProductName

$ResolvedArchiveDestination = ""
if ($ArchiveDestination) {
    $ResolvedArchiveDestination = Get-NormalizedPath -Path $ArchiveDestination
    if ((Split-Path -Leaf $ResolvedArchiveDestination) -cne $ProductName) {
        throw "归档目标目录名必须为 $ProductName。"
    }
    $ArchiveParent = Split-Path -Parent $ResolvedArchiveDestination
    if (-not (Test-Path -LiteralPath $ArchiveParent -PathType Container)) {
        throw "归档父目录不存在: $ArchiveParent"
    }
    if (Test-Path -LiteralPath $ResolvedArchiveDestination) {
        throw "归档目标已存在，拒绝覆盖: $ResolvedArchiveDestination"
    }
}

if (-not $OutputRoot) {
    $OutputRoot = $ReleaseArchiveRoot
}
$OutputRoot = Get-NormalizedPath -Path $OutputRoot -BasePath $ProjectRoot
if (-not (Test-PathWithin -Candidate $OutputRoot -Parent $ReleaseArchiveRoot -AllowEqual)) {
    throw "构建输出必须位于项目外层的「历史归档/发布版本」目录内: $OutputRoot"
}

$ResolvedPython = Resolve-AfcPython -RequestedPath $PythonPath
$BuildEnvironment = Get-BuildEnvironment -PythonExecutable $ResolvedPython
if ($BuildEnvironment.environment_name -ine "AFC") {
    throw "构建解释器不属于 AFC 环境: $($BuildEnvironment.python_prefix)"
}
if ($BuildEnvironment.packages.PyInstaller -ne $RequiredPyInstallerVersion) {
    throw "PyInstaller 版本必须为 $RequiredPyInstallerVersion，当前为 $($BuildEnvironment.packages.PyInstaller)。"
}

$GitCommit = (& git -C $ProjectRoot rev-parse HEAD) | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 提交。"
}
$GitCommit = $GitCommit.Trim()
$GitStatusLines = @(& git -C $ProjectRoot status --porcelain --untracked-files=normal)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 工作区状态。"
}
$GitDirty = $GitStatusLines.Count -gt 0
$GitState = if ($GitDirty) { "有未提交修改（测试构建）" } else { "干净" }
$BuildTime = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK", [System.Globalization.CultureInfo]::InvariantCulture)

if ($ResolvedArchiveDestination -and $GitDirty) {
    throw "工作区有未提交修改；允许测试构建，但禁止正式归档。"
}

New-Item -ItemType Directory -Path $ReleaseArchiveRoot -Force | Out-Null
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$FinalReleaseDirectory = Join-Path $OutputRoot $DebugDirectoryName
$FinalMinimalArchive = Join-Path $OutputRoot "$ProductName.zip"
$LegacyProductDirectory = Join-Path $OutputRoot $ProductName
$ReleaseBuildRoot = Join-Path $OutputRoot ".build-$ProductName"
$ReleasePayloadRootNames = @(
    "_internal",
    "发布说明.txt",
    "build-manifest.json",
    "SHA256SUMS.txt"
)

function Test-ReleasePayloadRootName {
    param([Parameter(Mandatory = $true)][string]$Name)

    if ($ReleasePayloadRootNames -icontains $Name) {
        return $true
    }
    if ($Name -match '^AFC.+\.exe$') {
        return $true
    }
    return $false
}
<#
候选构建可复用上次失败遗留的 PyInstaller work 缓存；正式归档或显式
-CleanBuild 时清空整个版本构建目录。每次构建验证成功后都会删除 work。
#>
$RequireCleanBuild = [bool]$CleanBuild -or [bool]$ResolvedArchiveDestination
if ($RequireCleanBuild -and (Test-Path -LiteralPath $ReleaseBuildRoot)) {
    Remove-SafeDirectory -Path $ReleaseBuildRoot -AllowedParent $OutputRoot
}
New-Item -ItemType Directory -Path $ReleaseBuildRoot -Force | Out-Null

$WorkPath = Join-Path $ReleaseBuildRoot "work"
$DistPath = Join-Path $ReleaseBuildRoot "dist"
if ((-not $RequireCleanBuild) -and (Test-Path -LiteralPath $DistPath)) {
    Remove-SafeDirectory -Path $DistPath -AllowedParent $ReleaseBuildRoot
}
New-Item -ItemType Directory -Path $WorkPath -Force | Out-Null
New-Item -ItemType Directory -Path $DistPath -Force | Out-Null

Write-Host "使用 Python: $ResolvedPython"
Write-Host "构建输出: $ReleaseBuildRoot"
Write-Host "构建缓存: $(if ($RequireCleanBuild) { '全量清理' } else { '复用 work 缓存' })"
Write-Host "开始 PyInstaller onedir/windowed 构建..."

<#
沿用同目录既有发布脚本已经验证过的方式：先进入项目根目录，再只向
PyInstaller 传递相对路径，避免 6.12 在中文 Windows 上重复转码绝对路径。
同时仅在子进程期间启用 Python UTF-8 模式，调用结束后恢复用户环境。
#>
$SpecArgument = "packaging\AFC2_onedir.spec"
$WorkArgument = (Get-PortableRelativePath -BasePath $ProjectRoot -TargetPath $WorkPath).Replace("/", "\")
$DistArgument = (Get-PortableRelativePath -BasePath $ProjectRoot -TargetPath $DistPath).Replace("/", "\")
$PreviousPythonUtf8 = $env:PYTHONUTF8
$PreviousProcessPath = $env:PATH
$AfcLibraryBin = Join-Path $BuildEnvironment.python_prefix "Library\bin"
if (-not (Test-Path -LiteralPath $AfcLibraryBin -PathType Container)) {
    throw "AFC 环境缺少 Conda 动态库目录: $AfcLibraryBin"
}
Push-Location -LiteralPath $ProjectRoot
try {
    $env:PYTHONUTF8 = "1"
    $env:PATH = $AfcLibraryBin + [System.IO.Path]::PathSeparator + $PreviousProcessPath
    & $ResolvedPython -m PyInstaller `
        --noconfirm `
        --log-level WARN `
        --workpath $WorkArgument `
        --distpath $DistArgument `
        $SpecArgument
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 构建失败，退出码: $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $PreviousPythonUtf8) {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONUTF8 = $PreviousPythonUtf8
    }
    $env:PATH = $PreviousProcessPath
    Pop-Location
}

$StagedReleaseDirectory = Join-Path $DistPath $ProductName
if (-not (Test-Path -LiteralPath $StagedReleaseDirectory -PathType Container)) {
    throw "未找到 PyInstaller 发布目录: $StagedReleaseDirectory"
}
$ReleaseDirectory = $StagedReleaseDirectory

$DependencyLines = @(
    "Python: $($BuildEnvironment.python_version)",
    "PyInstaller: $($BuildEnvironment.packages.PyInstaller)",
    "NumPy: $($BuildEnvironment.packages.numpy)",
    "Pandas: $($BuildEnvironment.packages.pandas)",
    "Matplotlib: $($BuildEnvironment.packages.matplotlib)",
    "scikit-learn（构建环境存在，发布包排除）: $($BuildEnvironment.packages.'scikit-learn')",
    "平台: $($BuildEnvironment.platform)"
)

$ReleaseNotes = [System.IO.File]::ReadAllText($ReleaseNotesTemplate, [System.Text.Encoding]::UTF8)
$ReleaseNotes = $ReleaseNotes.Replace("{{VERSION}}", $ProductName)
$ReleaseNotes = $ReleaseNotes.Replace("{{GIT_COMMIT}}", $GitCommit)
$ReleaseNotes = $ReleaseNotes.Replace("{{GIT_STATE}}", $GitState)
$ReleaseNotes = $ReleaseNotes.Replace("{{BUILD_TIME}}", $BuildTime)
$ReleaseNotes = $ReleaseNotes.Replace("{{DEPENDENCIES}}", ($DependencyLines -join [Environment]::NewLine))
Write-Utf8NoBom -Path (Join-Path $ReleaseDirectory "发布说明.txt") -Content $ReleaseNotes

$Manifest = [ordered]@{
    product = $ProductName
    git_commit = $GitCommit
    git_dirty = $GitDirty
    build_time = $BuildTime
    python_executable_name = [System.IO.Path]::GetFileName($BuildEnvironment.python_executable)
    python_prefix_name = [System.IO.Path]::GetFileName($BuildEnvironment.python_prefix)
    python_version = $BuildEnvironment.python_version
    platform = $BuildEnvironment.platform
    dependencies = [ordered]@{
        PyInstaller = $BuildEnvironment.packages.PyInstaller
        numpy = $BuildEnvironment.packages.numpy
        pandas = $BuildEnvironment.packages.pandas
        matplotlib = $BuildEnvironment.packages.matplotlib
        "scikit-learn_build_only" = $BuildEnvironment.packages.'scikit-learn'
    }
    checksum_file = "SHA256SUMS.txt"
}
$ManifestJson = $Manifest | ConvertTo-Json -Depth 6
Write-Utf8NoBom -Path (Join-Path $ReleaseDirectory "build-manifest.json") -Content ($ManifestJson + [Environment]::NewLine)

$ChecksumLines = New-Object System.Collections.Generic.List[string]
$PayloadFiles = Get-ChildItem -LiteralPath $ReleaseDirectory -Recurse -Force -File |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Sort-Object FullName
foreach ($file in $PayloadFiles) {
    $relativePath = Get-PortableRelativePath -BasePath $ReleaseDirectory -TargetPath $file.FullName
    $hash = Get-FileSha256WithRetry -Path $file.FullName
    $ChecksumLines.Add("$hash  $relativePath")
}
Write-Utf8NoBom -Path (Join-Path $ReleaseDirectory "SHA256SUMS.txt") -Content (($ChecksumLines -join [Environment]::NewLine) + [Environment]::NewLine)

& $VerificationScript `
    -ReleaseDirectory $ReleaseDirectory `
    -PythonPath $ResolvedPython `
    -ExpectedVersion $ProductName `
    -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "发布目录验证失败。"
}

# 最小 ZIP 只从尚未混入本机运行输入的候选目录生成。解压后再次执行
# 完整验证，确保实际交付的压缩包与已验证候选目录一致且可启动。
Add-Type -AssemblyName System.IO.Compression.FileSystem
$StagedMinimalArchive = Join-Path $ReleaseBuildRoot "$ProductName.zip"
$ZipVerificationDirectory = Join-Path $ReleaseBuildRoot "zip-verification"
if (Test-Path -LiteralPath $StagedMinimalArchive) {
    Remove-SafeFile -Path $StagedMinimalArchive -AllowedParent $ReleaseBuildRoot
}
if (Test-Path -LiteralPath $ZipVerificationDirectory) {
    Remove-SafeDirectory -Path $ZipVerificationDirectory -AllowedParent $ReleaseBuildRoot
}
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $ReleaseDirectory,
    $StagedMinimalArchive,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
[System.IO.Compression.ZipFile]::ExtractToDirectory(
    $StagedMinimalArchive,
    $ZipVerificationDirectory
)
& $VerificationScript `
    -ReleaseDirectory $ZipVerificationDirectory `
    -PythonPath $ResolvedPython `
    -ExpectedVersion $ProductName `
    -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "最小发布 ZIP 解压验证失败。"
}

$DebugStateBackupDirectory = Join-Path $ReleaseBuildRoot "debug-state-backup"
$DebugStateHashes = @{}
$DebugStateSourceDirectories = @($FinalReleaseDirectory, $LegacyProductDirectory) |
    Select-Object -Unique
foreach ($debugStateSourceDirectory in $DebugStateSourceDirectories) {
    if (Test-Path -LiteralPath $debugStateSourceDirectory -PathType Container) {
        New-Item -ItemType Directory -Path $DebugStateBackupDirectory -Force | Out-Null
        $DebugStateEntries = Get-ChildItem -LiteralPath $debugStateSourceDirectory -Force |
            Where-Object { -not (Test-ReleasePayloadRootName -Name $_.Name) }
        foreach ($debugStateEntry in $DebugStateEntries) {
            $backupPath = Join-Path $DebugStateBackupDirectory $debugStateEntry.Name
            if (Test-Path -LiteralPath $backupPath) {
                Write-Warning "同版本旧目录存在同名状态项，保留版本调试目录中的文件: $($debugStateEntry.Name)"
                continue
            }
            Copy-Item -LiteralPath $debugStateEntry.FullName -Destination $backupPath -Recurse -Force
            Write-Host "已备份调试状态: $($debugStateEntry.FullName)"
        }
    }
}
if (Test-Path -LiteralPath $DebugStateBackupDirectory -PathType Container) {
    $DebugStateFiles = Get-ChildItem -LiteralPath $DebugStateBackupDirectory -Recurse -Force -File
    foreach ($debugStateFile in $DebugStateFiles) {
        $relativePath = Get-PortableRelativePath `
            -BasePath $DebugStateBackupDirectory `
            -TargetPath $debugStateFile.FullName
        $DebugStateHashes[$relativePath] = Get-FileSha256WithRetry -Path $debugStateFile.FullName
    }
}

if (Test-Path -LiteralPath $FinalReleaseDirectory) {
    Remove-SafeDirectory -Path $FinalReleaseDirectory -AllowedParent $OutputRoot
}
[System.IO.Directory]::Move($StagedReleaseDirectory, $FinalReleaseDirectory)
$ReleaseDirectory = $FinalReleaseDirectory
if (Test-Path -LiteralPath $DebugStateBackupDirectory -PathType Container) {
    Get-ChildItem -LiteralPath $DebugStateBackupDirectory -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $ReleaseDirectory -Recurse -Force
    }
}
foreach ($relativePath in $DebugStateHashes.Keys) {
    $restoredPath = Join-Path $ReleaseDirectory $relativePath.Replace(
        "/",
        [System.IO.Path]::DirectorySeparatorChar
    )
    $restoredHash = Get-FileSha256WithRetry -Path $restoredPath
    if ($restoredHash -cne $DebugStateHashes[$relativePath]) {
        throw "调试状态恢复后 SHA-256 不一致: $relativePath"
    }
}
Write-Host "已原样恢复并校验 $($DebugStateHashes.Count) 个调试状态文件。"

# 临时目录自检通过后，再从最终发布目录原地启动一次。同步盘可能在目录
# 换入后短暂锁定新 EXE，这一步确保交付路径本身也能完成初始化。
$FinalExecutable = Join-Path $ReleaseDirectory "$ProductName.exe"
$FinalStartupLog = Join-Path $ReleaseDirectory "startup-error.log"
$FinalStartupLogExisted = Test-Path -LiteralPath $FinalStartupLog -PathType Leaf
$PreviousFinalSmokeFlag = $env:AFC_RELEASE_SMOKE_TEST
$PreviousFinalSuppressMessageboxes = $env:SUPPRESS_MESSAGEBOXES
try {
    $env:AFC_RELEASE_SMOKE_TEST = "1"
    $env:SUPPRESS_MESSAGEBOXES = "1"
    $FinalSmokeProcess = Start-Process `
        -FilePath $FinalExecutable `
        -WorkingDirectory $ReleaseDirectory `
        -WindowStyle Hidden `
        -PassThru
    if (-not $FinalSmokeProcess.WaitForExit(30000)) {
        Stop-Process -Id $FinalSmokeProcess.Id -Force -ErrorAction SilentlyContinue
        throw "最终发布目录原地启动自检超时。"
    }
    if ($FinalSmokeProcess.ExitCode -ne 0) {
        $FinalSmokeLogText = ""
        if (Test-Path -LiteralPath $FinalStartupLog -PathType Leaf) {
            $FinalSmokeLogText = [System.IO.File]::ReadAllText(
                $FinalStartupLog,
                [System.Text.Encoding]::UTF8
            )
        }
        throw "最终发布目录原地启动自检失败，退出码 $($FinalSmokeProcess.ExitCode)。`n$FinalSmokeLogText"
    }
    if ((-not $FinalStartupLogExisted) -and
        (Test-Path -LiteralPath $FinalStartupLog -PathType Leaf)) {
        Remove-SafeFile -Path $FinalStartupLog -AllowedParent $ReleaseDirectory
    }
}
finally {
    if ($null -eq $PreviousFinalSmokeFlag) {
        Remove-Item Env:AFC_RELEASE_SMOKE_TEST -ErrorAction SilentlyContinue
    }
    else {
        $env:AFC_RELEASE_SMOKE_TEST = $PreviousFinalSmokeFlag
    }
    if ($null -eq $PreviousFinalSuppressMessageboxes) {
        Remove-Item Env:SUPPRESS_MESSAGEBOXES -ErrorAction SilentlyContinue
    }
    else {
        $env:SUPPRESS_MESSAGEBOXES = $PreviousFinalSuppressMessageboxes
    }
}
Write-Host "最终发布目录原地启动自检通过: $FinalExecutable"

# 当前版本的新调试目录和 ZIP 都验证成功后，才替换同版本的最小包。
# 其他版本的调试目录和 ZIP 必须长期保留，不参与本次构建清理。
if (Test-Path -LiteralPath $FinalMinimalArchive) {
    Remove-SafeFile -Path $FinalMinimalArchive -AllowedParent $OutputRoot
}
[System.IO.File]::Move($StagedMinimalArchive, $FinalMinimalArchive)

if (Test-Path -LiteralPath $ReleaseBuildRoot) {
    Remove-SafeDirectory -Path $ReleaseBuildRoot -AllowedParent $OutputRoot
}

Write-Host "发布构建与验证通过: $ReleaseDirectory"
Write-Host "最小发布 ZIP: $FinalMinimalArchive"
Write-Host "已清理 PyInstaller 中间目录: $ReleaseBuildRoot"

if ($ResolvedArchiveDestination) {
    $ArchiveCommit = ((& git -C $ProjectRoot rev-parse HEAD) | Out-String).Trim()
    $ArchiveStatusLines = @(& git -C $ProjectRoot status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0 -or $ArchiveCommit -cne $GitCommit -or $ArchiveStatusLines.Count -gt 0) {
        throw "构建期间 Git 提交或工作区状态发生变化，拒绝正式归档。"
    }
    $ArchiveParent = Split-Path -Parent $ResolvedArchiveDestination
    $TemporaryArchive = Join-Path $ArchiveParent (".{0}.partial-{1}" -f $ProductName, [guid]::NewGuid().ToString("N"))
    try {
        [System.IO.Compression.ZipFile]::ExtractToDirectory(
            $FinalMinimalArchive,
            $TemporaryArchive
        )

        & $VerificationScript `
            -ReleaseDirectory $TemporaryArchive `
            -PythonPath $ResolvedPython `
            -ExpectedVersion $ProductName `
            -ProjectRoot $ProjectRoot
        if ($LASTEXITCODE -ne 0) {
            throw "临时归档副本验证失败。"
        }
        if (Test-Path -LiteralPath $ResolvedArchiveDestination) {
            throw "归档目标在复制期间被创建，拒绝覆盖: $ResolvedArchiveDestination"
        }
        [System.IO.Directory]::Move($TemporaryArchive, $ResolvedArchiveDestination)
        Write-Host "正式归档完成: $ResolvedArchiveDestination"
    }
    finally {
        if (Test-Path -LiteralPath $TemporaryArchive) {
            Remove-SafeDirectory -Path $TemporaryArchive -AllowedParent $ArchiveParent
        }
    }
}
else {
    Write-Host "未提供 -ArchiveDestination；本次已将构建验证产物保存到历史归档的发布版本目录。"
}
