param(
    [string]$DataPath = "data/slm_mit_vocabmix_ko_v1.jsonl",
    [string]$OutputDir = "artifacts_ko_mit_vocabmix_v3_clean",
    [string]$TokenizerModel = "artifacts_tokenizer_spm_ko_v1/ko_spm_16k.model",
    [string]$ResumeFrom = "artifacts_ondevice_termlogic_v2_1200/slm_best.pt",
    [int]$SeqLen = 384,
    [int]$BatchSize = 12,
    [int]$GradAccumSteps = 2,
    [int]$Steps = 2200,
    [int]$EvalInterval = 100,
    [int]$SaveInterval = 250,
    [int]$EvalBatches = 24,
    [double]$Lr = 6.0e-5,
    [double]$WeightDecay = 0.1,
    [double]$GradClip = 1.0,
    [double]$ValRatio = 0.02,
    [int]$WarmupSteps = 120,
    [int]$DModel = 384,
    [int]$NHeads = 6,
    [int]$NLayers = 8,
    [int]$MlpMult = 4,
    [double]$Dropout = 0.1,
    [double]$KoFocus = 2.4,
    [double]$DocFocus = 0.15,
    [double]$EmaDecay = 0.995,
    [double]$ConsistencyLambda = 0.04,
    [int]$NumWorkers = 0,
    [switch]$RebuildDataset,
    [double]$TargetKoRatio = 0.84,
    [int]$MaxTotalRows = 180000,
    [int]$MaxKeptRowsPerFile = 70000,
    [int]$MaxRowsPerSourceDataset = 0,
    [double]$KoMinOutputHangulRatio = 0.28,
    [double]$KoMaxOutputLatinRatio = 0.60,
    [int]$MaxKoPrefixRepeat = 300,
    [switch]$PlainSft,
    [switch]$NoResume,
    [switch]$TurboMode,
    [int]$TurboMinSeqLen = 128,
    [double]$TurboSeqWarmupRatio = 0.55,
    [int]$TurboMaxFrozenLayers = 3,
    [double]$TurboDepthWarmupRatio = 0.35,
    [switch]$TurboFreezeEmbeddings,
    [int]$Seed = 42,
    [string]$PythonExe = ""
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

function Resolve-PythonExe([string]$repoRoot) {
    if ($PythonExe -and (Test-PythonExe -exePath $PythonExe)) {
        return $PythonExe
    }

    $localVenv = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $candidates = @(
        $localVenv
    )
    foreach ($c in $candidates) {
        if (Test-PythonExe -exePath $c) {
            return $c
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonExe -exePath $cmd.Source)) {
        return $cmd.Source
    }
    if (Test-PythonExe -exePath "C:\Program Files\Python311\python.exe") {
        return "C:\Program Files\Python311\python.exe"
    }
    throw "No usable python executable found."
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

& $pythonExe -c "import torch, sentencepiece; print('ok')" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Python interpreter '$pythonExe' cannot import required modules (torch, sentencepiece). Use -PythonExe or install packages first."
}

if ($RebuildDataset) {
    Write-Output "==> Rebuilding MIT+Korean-vocab dataset"
    & $pythonExe scripts/build_mit_vocabmix_ko_v1.py `
        --add_vocab_rows `
        --shuffle `
        --target_ko_ratio $TargetKoRatio `
        --max_total_rows $MaxTotalRows `
        --max_kept_rows_per_file $MaxKeptRowsPerFile `
        --max_rows_per_source_dataset $MaxRowsPerSourceDataset `
        --ko_min_output_hangul_ratio $KoMinOutputHangulRatio `
        --ko_max_output_latin_ratio $KoMaxOutputLatinRatio `
        --max_ko_prefix_repeat $MaxKoPrefixRepeat
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path $DataPath)) {
    throw "data path not found: $DataPath"
}
if (-not (Test-Path $TokenizerModel)) {
    throw "tokenizer model not found: $TokenizerModel"
}

$cmd = @(
    "scripts/train_slm.py",
    "--data_path", $DataPath,
    "--output_dir", $OutputDir,
    "--tokenizer_type", "spm",
    "--tokenizer_model", $TokenizerModel,
    "--seq_len", "$SeqLen",
    "--batch_size", "$BatchSize",
    "--grad_accum_steps", "$GradAccumSteps",
    "--steps", "$Steps",
    "--eval_interval", "$EvalInterval",
    "--save_interval", "$SaveInterval",
    "--eval_batches", "$EvalBatches",
    "--lr", "$Lr",
    "--weight_decay", "$WeightDecay",
    "--grad_clip", "$GradClip",
    "--val_ratio", "$ValRatio",
    "--warmup_steps", "$WarmupSteps",
    "--d_model", "$DModel",
    "--n_heads", "$NHeads",
    "--n_layers", "$NLayers",
    "--mlp_mult", "$MlpMult",
    "--dropout", "$Dropout",
    "--ko_focus", "$KoFocus",
    "--doc_focus", "$DocFocus",
    "--ema_decay", "$EmaDecay",
    "--consistency_lambda", "$ConsistencyLambda",
    "--num_workers", "$NumWorkers",
    "--seed", "$Seed",
    "--sample_prompt", "### Instruction`nSay hello in Korean in one sentence.`n`n### Response`n",
    "--save_step_checkpoints"
)

if (-not $NoResume -and $ResumeFrom -and (Test-Path $ResumeFrom)) {
    $cmd += @("--resume_from", $ResumeFrom, "--resume_weights_only", "--reset_best_on_resume")
}
if ($PlainSft) {
    $cmd += "--plain_sft"
}
if ($TurboMode) {
    $cmd += @(
        "--turbo_mode",
        "--turbo_min_seq_len", "$TurboMinSeqLen",
        "--turbo_seq_warmup_ratio", "$TurboSeqWarmupRatio",
        "--turbo_max_frozen_layers", "$TurboMaxFrozenLayers",
        "--turbo_depth_warmup_ratio", "$TurboDepthWarmupRatio"
    )
    if ($TurboFreezeEmbeddings) {
        $cmd += "--turbo_freeze_embeddings"
    }
}

Write-Output "==> Train with MIT+Korean-vocab mix"
& $pythonExe @cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "==> Done"

