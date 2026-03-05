param(
    [string]$BaseCheckpoint = "artifacts_ondevice_best/slm_ondevice_fp16.pt",
    [string]$SpecPath = "data/ccl_specbook_v1.jsonl",
    [string]$DataPath = "data/slm_mit_unified_v4.jsonl",
    [string]$OutputDir = "artifacts_omega3_holographic",
    [int]$Steps = 80,
    [int]$MaxSpecs = 1200,
    [int]$BatchSize = 24,
    [string]$Layers = "",
    [int]$MemoryTopK = 6,
    [double]$BaseAlpha = 1.0,
    [double]$SoftThreshold = 0.42,
    [int]$MezoIters = 3,
    [double]$MezoSigma = 0.12,
    [double]$MezoLr = 0.25,
    [int]$SaveInterval = 4,
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
    "scripts/run_omega3_holographic.py",
    "--base_checkpoint", $BaseCheckpoint,
    "--spec_path", $SpecPath,
    "--data_path", $DataPath,
    "--output_dir", $OutputDir,
    "--steps", $Steps,
    "--max_specs", $MaxSpecs,
    "--batch_size", $BatchSize,
    "--memory_top_k", $MemoryTopK,
    "--base_alpha", $BaseAlpha,
    "--soft_threshold", $SoftThreshold,
    "--mezo_iters", $MezoIters,
    "--mezo_sigma", $MezoSigma,
    "--mezo_lr", $MezoLr,
    "--save_interval", $SaveInterval,
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
