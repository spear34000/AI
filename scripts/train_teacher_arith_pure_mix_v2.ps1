param(
    [switch]$StartDetached,
    [string]$RunTag = "runA",
    [string]$ResumeFrom = ""
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

function Quote-PSArg([string]$Value) {
    if ($null -eq $Value) { return "''" }
    return "'" + ($Value -replace "'", "''") + "'"
}

$py = Get-TrainPython
$tokenDir = Join-Path $root 'artifacts_tokenizer_spm_from_stage2_v1'
$spmModel = Join-Path $tokenDir 'spm16k.model'
$trainData = Join-Path $root 'data\teacher_arith_pure_mix_v2_clean_train.jsonl'
$evalData = Join-Path $root 'data\teacher_arith_pure_mix_v2_clean_eval.jsonl'
$outDir = Join-Path $root ("artifacts_teacher_arith_pure_mix_fromscratch_v2_clean_{0}" -f $RunTag)
$logDir = Join-Path $root 'logs'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir ("arith_pure_mix_v2_clean_{0}.log" -f $stamp)
$errLogFile = [IO.Path]::ChangeExtension($logFile, '.err.log')

New-Item -ItemType Directory -Force -Path $tokenDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ($StartDetached) {
    if ((-not (Test-Path $spmModel)) -or (-not (Test-Path $trainData)) -or (-not (Test-Path $evalData))) {
        throw "StartDetached requires prebuilt tokenizer and dataset files"
    }
    $detachedArgs = @(
        '-u', 'scripts/train_slm.py',
        '--data_path', 'data/teacher_arith_pure_mix_v2_clean_train.jsonl',
        '--output_dir', $outDir,
        '--steps', '25000',
        '--batch_size', '2',
        '--grad_accum_steps', '8',
        '--seq_len', '512',
        '--lr', '1.5e-4',
        '--weight_decay', '0.1',
        '--warmup_steps', '400',
        '--grad_clip', '1.0',
        '--val_ratio', '0.02',
        '--eval_interval', '200',
        '--save_interval', '1000',
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
        '--turbo_min_seq_len', '192',
        '--turbo_seq_warmup_ratio', '0.35',
        '--turbo_max_frozen_layers', '4',
        '--turbo_depth_warmup_ratio', '0.25',
        '--ema_decay', '0.0',
        '--consistency_lambda', '0.0',
        '--plain_sft',
        '--response_loss_only'
    )
    if ([string]::IsNullOrWhiteSpace($ResumeFrom) -eq $false) {
        $detachedArgs += @('--resume_from', $ResumeFrom)
    }
    $proc = Start-Process -FilePath $py `
        -ArgumentList $detachedArgs `
        -WorkingDirectory $root `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $errLogFile `
        -PassThru
    [pscustomobject]@{
        pid = $proc.Id
        log = $logFile
        err = $errLogFile
        output_dir = $outDir
        train_data = $trainData
        resume_from = $ResumeFrom
    } | ConvertTo-Json -Compress
    exit 0
}

if (-not (Test-Path $spmModel)) {
    & $py scripts/export_spm_from_checkpoint.py `
        --checkpoint artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt `
        --out_model $spmModel
    if ($LASTEXITCODE -ne 0) {
        throw "failed to export spm model"
    }
}

if ((-not (Test-Path $trainData)) -or (-not (Test-Path $evalData))) {
    & $py scripts/build_teacher_arith_pure_mix_v2.py
    if ($LASTEXITCODE -ne 0) {
        throw "dataset build failed"
    }
}

$cmd = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/teacher_arith_pure_mix_v2_clean_train.jsonl',
    '--output_dir', $outDir,
    '--steps', '25000',
    '--batch_size', '2',
    '--grad_accum_steps', '8',
    '--seq_len', '512',
    '--lr', '1.5e-4',
    '--weight_decay', '0.1',
    '--warmup_steps', '400',
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', '200',
    '--save_interval', '1000',
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
    '--turbo_min_seq_len', '192',
    '--turbo_seq_warmup_ratio', '0.35',
    '--turbo_max_frozen_layers', '4',
    '--turbo_depth_warmup_ratio', '0.25',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0',
    '--plain_sft',
    '--response_loss_only'
)

if ([string]::IsNullOrWhiteSpace($ResumeFrom) -eq $false) {
    $cmd += @('--resume_from', $ResumeFrom)
}

& $py @cmd
if ($LASTEXITCODE -ne 0) {
    throw "train_slm.py failed with exit code $LASTEXITCODE"
}
