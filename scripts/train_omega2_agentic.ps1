param(
    [string]$BaseCheckpoint = "artifacts_ondevice_best/slm_ondevice_fp16.pt",
    [string]$SpecPath = "data/ccl_specbook_v1.jsonl",
    [string]$DataPath = "data/slm_mit_unified_v4.jsonl",
    [string]$OutputDir = "artifacts_omega2_agentic",
    [int]$Steps = 80,
    [int]$MaxSpecs = 1200,
    [int]$GenerateBatch = 24,
    [int]$PatchTopK = 4,
    [int]$SearchBranches = 3,
    [string]$SearchTemps = "0.0,0.6,0.9",
    [string]$SearchTopK = "0,60,120",
    [string]$SearchTopP = "1.0,0.92,0.97",
    [double]$SoftThreshold = 0.42,
    [double]$NoveltyThreshold = 0.90,
    [int]$DistillTriggerPatches = 1000,
    [int]$DistillCooldownSteps = 24,
    [int]$DistillMaxRows = 2000,
    [switch]$AutoDistill,
    [string]$DistillOutputDir = "artifacts_omega2_distill",
    [int]$DistillSteps = 160,
    [int]$DistillBatchSize = 24,
    [switch]$FlushAfterDistill,
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
    "scripts/run_omega2_agentic.py",
    "--base_checkpoint", $BaseCheckpoint,
    "--spec_path", $SpecPath,
    "--data_path", $DataPath,
    "--output_dir", $OutputDir,
    "--steps", $Steps,
    "--max_specs", $MaxSpecs,
    "--generate_batch", $GenerateBatch,
    "--patch_top_k", $PatchTopK,
    "--search_branches", $SearchBranches,
    "--search_temps", $SearchTemps,
    "--search_top_k", $SearchTopK,
    "--search_top_p", $SearchTopP,
    "--soft_threshold", $SoftThreshold,
    "--novelty_threshold", $NoveltyThreshold,
    "--distill_trigger_patches", $DistillTriggerPatches,
    "--distill_cooldown_steps", $DistillCooldownSteps,
    "--distill_max_rows", $DistillMaxRows,
    "--distill_output_dir", $DistillOutputDir,
    "--distill_steps", $DistillSteps,
    "--distill_batch_size", $DistillBatchSize,
    "--save_interval", $SaveInterval,
    "--seed", $Seed,
    "--device", $Device
)

if ($AutoDistill) {
    $argsList += @("--auto_distill")
}
if ($FlushAfterDistill) {
    $argsList += @("--flush_after_distill")
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

