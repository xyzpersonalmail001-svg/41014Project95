param(
    [string]$PythonPath = '',
    [string]$GitPath = '',
    [switch]$IncludeTutorial
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

# 临时文件也放在项目内，避免桌面沙箱的系统临时目录权限问题。
$projectTemp = Join-Path $projectRoot '.tmp'
New-Item -ItemType Directory -Path $projectTemp -Force | Out-Null
$env:TEMP = $projectTemp
$env:TMP = $projectTemp

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (-not $PythonPath) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundledPython) { $PythonPath = $bundledPython }
        else {
            $candidate = Get-Command python -ErrorAction SilentlyContinue
            if ($candidate -and $candidate.Source -notlike '*\WindowsApps\*') { $PythonPath = $candidate.Source }
        }
    }
    if (-not $PythonPath) { throw 'Supply -PythonPath with an installed 64-bit Python 3.12 executable.' }
    & $PythonPath -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for this verified environment'"
    if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
    & $PythonPath -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

& '.\.venv\Scripts\python.exe' -c "import sys,struct; assert sys.platform == 'win32' and sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Verified target: Windows x64 / Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Existing/new virtual environment does not match the supported target.' }

& '.\.venv\Scripts\python.exe' -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

if (-not $GitPath) {
    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCommand) { $GitPath = $gitCommand.Source }
    else {
        $bundledGit = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
        if (Test-Path -LiteralPath $bundledGit) { $GitPath = $bundledGit }
    }
}
if (-not $GitPath) { throw 'Supply -GitPath with a Git executable.' }

$resources = Get-Content -LiteralPath resources.lock.json -Raw | ConvertFrom-Json
foreach ($item in $resources.PSObject.Properties) {
    # The UR3 tutorial is a reference, not a TurtleBot runtime dependency.
    if ($item.Name -eq 'teacher_tutorial' -and -not $IncludeTutorial) { continue }
    $resource = $item.Value
    $destination = Join-Path $projectRoot $resource.directory
    if (Test-Path -LiteralPath $destination) {
        $commit = & $GitPath -C $destination rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $commit -ne $resource.commit) {
            throw "Existing resource has another revision: $destination. It was left unchanged."
        }
        $localChanges = & $GitPath -C $destination status --porcelain
        if ($LASTEXITCODE -ne 0 -or $localChanges) {
            throw "Existing resource has local changes: $destination. It was left unchanged."
        }
        continue
    }
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    & $GitPath -C $destination init --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Resource repository initialization failed.' }
    & $GitPath -C $destination remote add origin $resource.url
    if ($LASTEXITCODE -ne 0) { throw 'Adding resource remote failed.' }
    if ($resource.sparse_paths.Count -gt 0) {
        & $GitPath -C $destination sparse-checkout set @($resource.sparse_paths)
        if ($LASTEXITCODE -ne 0) { throw 'Sparse checkout configuration failed.' }
    }
    & $GitPath -C $destination fetch --depth 1 --filter=blob:none origin $resource.commit
    if ($LASTEXITCODE -ne 0) { throw 'Resource download failed.' }
    & $GitPath -C $destination checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Resource checkout failed.' }
}
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& '.\.venv\Scripts\python.exe' -c "from evaluation.common.release_support import verify_assets; print(verify_assets())"
if ($LASTEXITCODE -ne 0) { throw 'Official model asset hash check failed.' }
Write-Host 'Setup complete. Run: .\.venv\Scripts\python.exe run_demo.py --case all'
Write-Host 'Full task: .\.venv\Scripts\python.exe run_demo.py --case normal'
