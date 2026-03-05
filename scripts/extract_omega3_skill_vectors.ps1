param(
    [string]$Checkpoint = "artifacts_ondevice_best/slm_ondevice_fp16.pt",
    [string]$DataPath = "data/pure_ko_seed_v1.jsonl",
    [string]$MemoryPath = "artifacts_omega3_holographic/holo_memory.pt",
    [int]$MaxRows = 800,
    [int]$MaxPerGroup = 120,
    [string]$Layers = "",
    [ValidateSet("mean", "pca")]
    [string]$VectorMode = "pca",
    [int]$Seed = 42,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Test-PythonWithTorch([string]$exePath) {
    if (-not (Test-PythonExe -exePath $exePath)) {
        return $false
    }
    try {
        & $exePath -c "import torch; print(torch.__version__)" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Resolve-PythonExe([string]$repoRoot) {
    $localVenv = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $candidates = @(
        $localVenv,
        "C:\Program Files\Python311\python.exe"
    )

    foreach ($c in $candidates) {
        if (Test-PythonWithTorch -exePath $c) {
            return $c
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonWithTorch -exePath $cmd.Source)) {
        return $cmd.Source
    }

    throw "No usable python executable with torch found."
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

$argsList = @(
    "scripts/extract_omega3_skill_vectors.py",
    "--checkpoint", $Checkpoint,
    "--data_path", $DataPath,
    "--memory_path", $MemoryPath,
    "--max_rows", $MaxRows,
    "--max_per_group", $MaxPerGroup,
    "--vector_mode", $VectorMode,
    "--seed", $Seed,
    "--device", $Device
)
if ($Layers -and $Layers.Trim()) {
    $argsList += @("--layers", $Layers)
}

Push-Location $repoRoot
try {
    & $pythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
