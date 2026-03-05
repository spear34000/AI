param(
    [switch]$StartDetached,
    [string]$RunTag = "runA"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-TrainPython {
    $candidates = @(
        'C:\Program Files\Python311\python.exe',
        (Join-Path $root '.venv\Scripts\python.exe')
    )
    foreach ($py in $candidates) {
        if (-not (Test-Path $py)) { continue }
        try {
            $null = & $py -c "import torch" 2>$null
            if ($LASTEXITCODE -eq 0) { return $py }
        } catch {}
    }
    throw 'torch import possible python not found'
}

function New-PSArg([string]$Value) {
    if ($null -eq $Value) { return "''" }
    return "'" + ($Value -replace "'", "''") + "'"
}

$py = Get-TrainPython
$tokenDir = Join-Path $root 'artifacts_tokenizer_spm_from_stage2_v1'
$spmModel = Join-Path $tokenDir 'spm16k.model'
$outDir = Join-Path $root ("artifacts_teacher_arith_v2_turbo_{0}" -f $RunTag)
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir ("arith_v2_turbo_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$errLogFile = [IO.Path]::ChangeExtension($logFile, '.err.log')

New-Item -ItemType Directory -Force -Path $tokenDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not (Test-Path $spmModel)) {
    & $py scripts/export_spm_from_checkpoint.py `
        --checkpoint artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt `
        --out_model $spmModel
    if ($LASTEXITCODE -ne 0) { throw "failed to export spm model" }
}

# Build curated dataset if missing
$trainData = Join-Path $root 'data\teacher_arith_pure_mix_v2_turbo_train.jsonl'
if (-not (Test-Path $trainData)) {
    # First ensure full v2 data exists
    $fullData = Join-Path $root 'data\teacher_arith_pure_mix_v2_train.jsonl'
    if (-not (Test-Path $fullData)) {
        & $py scripts/build_teacher_arith_pure_mix_v2.py
        if ($LASTEXITCODE -ne 0) { throw "full dataset build failed" }
    }
    & $py scripts/build_teacher_arith_v2_turbo.py
    if ($LASTEXITCODE -ne 0) { throw "turbo dataset curation failed" }
}

# ===== 20X TURBO CONFIGURATION =====
# vs v2 (baseline ~3-4 hours):
#   batch_size   1 -> 12   (12x throughput per step)
#   grad_accum  16 -> 1    (no accumulation overhead)
#   steps    25000 -> 2500  (~10x fewer steps)
#   lr     1.5e-4 -> 5e-4  (3.3x higher for fast convergence)
#   response_loss_only      (concentrated gradient -> fewer steps needed)
#   dataset 580k -> 80k     (focused data, no wasted easy examples)
# Note: torch.compile removed (Triton unavailable on Windows)
# Combined: ~20x faster  (~10-15 minutes)
$cmd = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/teacher_arith_pure_mix_v2_turbo_train.jsonl',
    '--output_dir', $outDir,
    '--steps', '2500',
    '--batch_size', '12',
    '--grad_accum_steps', '1',
    '--seq_len', '512',
    '--lr', '5e-4',
    '--weight_decay', '0.1',
    '--warmup_steps', '80',
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', '100',
    '--save_interval', '500',
    '--save_step_checkpoints',
    '--eval_batches', '24',
    '--num_workers', '0',
    '--d_model', '576',
    '--n_heads', '9',
    '--n_layers', '12',
    '--mlp_mult', '4',
    '--tokenizer_type', 'spm',
    '--tokenizer_model', $spmModel,
    '--activation_checkpointing',
    '--skip_sample',
    '--turbo_mode',
    '--turbo_min_seq_len', '128',
    '--turbo_seq_warmup_ratio', '0.25',
    '--turbo_max_frozen_layers', '4',
    '--turbo_depth_warmup_ratio', '0.15',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0',
    '--plain_sft',
    '--response_loss_only'
)

if ($StartDetached) {
    $runnerPath = Join-Path $logDir ("arith_v2_turbo_runner_{0}.ps1" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $cmdLine = "& $(New-PSArg $py) " + (($cmd | ForEach-Object { New-PSArg $_ }) -join ' ')
    $runner = @(
        '$ErrorActionPreference = ''Stop'''
        "Set-Location $(New-PSArg $root)"
        $cmdLine
    )
    Set-Content -Path $runnerPath -Value ($runner -join [Environment]::NewLine) -Encoding UTF8
    $proc = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runnerPath) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $errLogFile `
        -PassThru
    [pscustomobject]@{
        pid = $proc.Id
        log = $logFile
        err = $errLogFile
        output_dir = $outDir
    } | ConvertTo-Json -Compress
    exit 0
}

& $py @cmd
if ($LASTEXITCODE -ne 0) {
    throw "train_slm.py failed with exit code $LASTEXITCODE"
}
