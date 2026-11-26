param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$CondaExe = 'conda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$destination = [IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$env:TEMP = Join-Path $destination 'temp'
$env:TMP = $env:TEMP
$env:PYINSTALLER_CONFIG_DIR = Join-Path $destination 'pyinstaller_cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location $projectRoot
try {
    & $CondaExe run --no-capture-output -n AFC python -m PyInstaller --noconfirm `
        --distpath (Join-Path $destination 'candidate') `
        --workpath (Join-Path $destination 'build') packaging/interval_tool.spec
    if ($LASTEXITCODE -ne 0) { throw 'Interval tool candidate build failed' }
} finally {
    Pop-Location
}
# Builds an isolated candidate only. Delivery ZIP requires runtime verification.
