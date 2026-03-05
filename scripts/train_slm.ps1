param(
    [string]$DataPath = "data/slm_mit_unified_v4.jsonl",
    [string]$OutputDir = "artifacts_slm_mit_unified_v4",
    [ValidateSet("byte", "spm")]
    [string]$TokenizerType = "byte",
    [string]$TokenizerModel = "",
    [int]$Steps = 300,
    [int]$BatchSize = 18,
    [int]$GradAccumSteps = 1,
    [int]$SeqLen = 384,
    [int]$DModel = 384,
    [int]$NHeads = 6,
    [int]$NLayers = 8,
    [int]$MlpMult = 4,
    [double]$Dropout = 0.1,
    [int]$NumWorkers = 0,
    [double]$LR = 3e-4,
    [double]$WeightDecay = 0.1,
    [int]$WarmupSteps = 30,
    [double]$GradClip = 1.0,
    [double]$ValRatio = 0.02,
    [double]$KoFocus = 1.0,
    [double]$DocFocus = 0.4,
    [double]$EMADecay = 0.995,
    [double]$ConsistencyLambda = 0.05,
    [double]$ConsistencyTemp = 1.5,
    [int]$ConsistencyWarmup = 40,
    [int]$EvalInterval = 50,
    [int]$SaveInterval = 100,
    [int]$EvalBatches = 20,
    [int]$MaxRecords = 0,
    [string]$ResumeFrom = "",
    [switch]$ResumeWeightsOnly,
    [string]$SamplePrompt = "### Instruction`nSay hello in Korean.`n`n### Response`n",
    [switch]$EvalWithEMA,
    [switch]$ActivationCheckpointing,
    [switch]$SaveStepCheckpoints,
    [switch]$SkipSample,
    [switch]$NoBestCheckpoint,
    [switch]$TurboMode,
    [int]$TurboMinSeqLen = 128,
    [double]$TurboSeqWarmupRatio = 0.55,
    [int]$TurboMaxFrozenLayers = 4,
    [double]$TurboDepthWarmupRatio = 0.35,
    [switch]$TurboFreezeEmbeddings
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

$argsList = @(
    "scripts/train_slm.py",
    "--data_path", $DataPath,
    "--output_dir", $OutputDir,
    "--tokenizer_type", $TokenizerType,
    "--steps", $Steps,
    "--batch_size", $BatchSize,
    "--grad_accum_steps", $GradAccumSteps,
    "--seq_len", $SeqLen,
    "--d_model", $DModel,
    "--n_heads", $NHeads,
    "--n_layers", $NLayers,
    "--mlp_mult", $MlpMult,
    "--dropout", $Dropout,
    "--num_workers", $NumWorkers,
    "--lr", $LR,
    "--weight_decay", $WeightDecay,
    "--warmup_steps", $WarmupSteps,
    "--grad_clip", $GradClip,
    "--val_ratio", $ValRatio,
    "--ko_focus", $KoFocus,
    "--doc_focus", $DocFocus,
    "--ema_decay", $EMADecay,
    "--consistency_lambda", $ConsistencyLambda,
    "--consistency_temp", $ConsistencyTemp,
    "--consistency_warmup", $ConsistencyWarmup,
    "--eval_interval", $EvalInterval,
    "--save_interval", $SaveInterval,
    "--eval_batches", $EvalBatches,
    "--max_records", $MaxRecords,
    "--sample_prompt", $SamplePrompt,
    "--turbo_min_seq_len", $TurboMinSeqLen,
    "--turbo_seq_warmup_ratio", $TurboSeqWarmupRatio,
    "--turbo_max_frozen_layers", $TurboMaxFrozenLayers,
    "--turbo_depth_warmup_ratio", $TurboDepthWarmupRatio
)

if ($TokenizerType -eq "spm") {
    if (-not $TokenizerModel -or $TokenizerModel.Trim().Length -eq 0) {
        throw "TokenizerModel is required when TokenizerType=spm"
    }
    $argsList += @("--tokenizer_model", $TokenizerModel)
}

if ($ResumeFrom -and $ResumeFrom.Trim().Length -gt 0) {
    $argsList += @("--resume_from", $ResumeFrom)
}
if ($ResumeWeightsOnly) {
    $argsList += @("--resume_weights_only")
}
if ($EvalWithEMA) {
    $argsList += @("--eval_with_ema")
}
if ($ActivationCheckpointing) {
    $argsList += @("--activation_checkpointing")
}
if ($SaveStepCheckpoints) {
    $argsList += @("--save_step_checkpoints")
}
if ($SkipSample) {
    $argsList += @("--skip_sample")
}
if ($NoBestCheckpoint) {
    $argsList += @("--no_best_checkpoint")
}
if ($TurboMode) {
    $argsList += @("--turbo_mode")
}
if ($TurboFreezeEmbeddings) {
    $argsList += @("--turbo_freeze_embeddings")
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
