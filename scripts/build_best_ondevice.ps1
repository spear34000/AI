param(
    [string]$OutputDir = "artifacts_ondevice_best",
    [ValidateSet("fp16", "bf16")]
    [string]$DType = "fp16",
    [int]$ProbeTopN = 2,
    [int]$ProbeMaxNewTokens = 72,
    [switch]$SkipProbe,
    [switch]$NoPreferEMA
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath, [bool]$RequireTorch = $false) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        if ($RequireTorch) {
            & $exePath -c "import torch" *> $null
            if ($LASTEXITCODE -ne 0) {
                return $false
            }
        }
        return $true
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
        if (Test-PythonExe -exePath $c -RequireTorch $true) {
            return $c
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonExe -exePath $cmd.Source -RequireTorch $true)) {
        return $cmd.Source
    }

    throw "No usable python executable found."
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

Push-Location $repoRoot
try {
    $argsList = @(
        "scripts/build_best_ondevice.py",
        "--output_dir", $OutputDir,
        "--dtype", $DType,
        "--probe_top_n", $ProbeTopN,
        "--probe_max_new_tokens", $ProbeMaxNewTokens
    )

    if ($SkipProbe) {
        $argsList += @("--skip_probe")
    }
    if ($NoPreferEMA) {
        $argsList += @("--no_prefer_ema")
    }

    & $pythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
