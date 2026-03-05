param(
    [string]$ResumeCheckpoint = "artifacts_upgrade_126m_v1_train/slm_best.pt",
    [string]$TrainPath = "data/final_datasets_corpus_v1.jsonl",
    [string]$OutDir = "artifacts_upgrade_126m_long_turbo_v1",
    [int]$Steps = 5200,
    [int]$SeqLen = 512,
    [int]$BatchSize = 1,
    [int]$GradAccumSteps = 16,
    [double]$LR = 1.0e-5,
    [int]$WarmupSteps = 120,
    [int]$EvalInterval = 100,
    [int]$SaveInterval = 200,
    [int]$EvalBatches = 8,
    [switch]$NoEval
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent

function Test-PythonExe([string]$exePath, [bool]$RequireTorch = $false) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($RequireTorch) {
            & $exePath -c "import torch" *> $null
            if ($LASTEXITCODE -ne 0) { return $false }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Resolve-PythonExe([string]$root) {
    $localVenv = Join-Path $root ".venv\Scripts\python.exe"
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

$pythonExe = Resolve-PythonExe -root $repoRoot

if (-not (Test-Path (Join-Path $repoRoot $ResumeCheckpoint))) {
    throw "resume checkpoint not found: $ResumeCheckpoint"
}
if (-not (Test-Path (Join-Path $repoRoot $TrainPath))) {
    throw "train dataset not found: $TrainPath"
}

Write-Output "==> Long Turbo continue training (126M)"
& (Join-Path $PSScriptRoot "train_slm.ps1") `
    -DataPath $TrainPath `
    -OutputDir $OutDir `
    -Steps $Steps `
    -BatchSize $BatchSize `
    -GradAccumSteps $GradAccumSteps `
    -SeqLen $SeqLen `
    -LR $LR `
    -WarmupSteps $WarmupSteps `
    -WeightDecay 0.1 `
    -GradClip 1.0 `
    -ValRatio 0.02 `
    -EMADecay 0.0 `
    -ConsistencyLambda 0.0 `
    -ConsistencyTemp 1.5 `
    -ConsistencyWarmup 1000000 `
    -EvalInterval $EvalInterval `
    -SaveInterval $SaveInterval `
    -EvalBatches $EvalBatches `
    -NumWorkers 0 `
    -ResumeFrom $ResumeCheckpoint `
    -ResumeWeightsOnly `
    -ActivationCheckpointing `
    -SkipSample `
    -TurboMode `
    -TurboMinSeqLen 256 `
    -TurboSeqWarmupRatio 0.40 `
    -TurboMaxFrozenLayers 6 `
    -TurboDepthWarmupRatio 0.30 `
    -TurboFreezeEmbeddings
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $NoEval) {
    $bestPath = Join-Path $OutDir "slm_best.pt"
    if (-not (Test-Path (Join-Path $repoRoot $bestPath))) {
        $bestPath = Join-Path $OutDir "slm_last.pt"
    }

    Write-Output "==> Eval upgraded long-turbo checkpoint"
    Push-Location $repoRoot
    try {
        & $pythonExe scripts/eval_ko_quality5x.py `
            --candidate_checkpoint $bestPath `
            --device cuda `
            --tool_cache_path data/tool_knowledge_cache_v3_clean.jsonl `
            --out_json (Join-Path $OutDir "eval_ko_quality5x.json")
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }
}
