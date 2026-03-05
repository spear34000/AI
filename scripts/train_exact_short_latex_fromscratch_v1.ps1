param(
    [switch]$Smoke,
    [switch]$StartDetached,
    [string]$RunTag = ''
)

$ErrorActionPreference = 'Stop'
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
    throw 'torch import 가능한 python을 찾지 못했습니다.'
}

function Quote-PSArg([string]$Value) {
    if ($null -eq $Value) { return "''" }
    return "'" + ($Value -replace "'", "''") + "'"
}

$py = Get-TrainPython
$tokenDir = Join-Path $root 'artifacts_tokenizer_spm_from_stage2_v1'
$spmModel = Join-Path $tokenDir 'spm16k.model'
$baseName = 'artifacts_exact_short_latex_50m_fromscratch_v1'
if ($RunTag) {
    $baseName = "${baseName}_$RunTag"
}
$outDir = Join-Path $root $baseName
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir ("exact_short_latex_fromscratch_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$errLogFile = [IO.Path]::ChangeExtension($logFile, '.err.log')

New-Item -ItemType Directory -Force -Path $tokenDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $spmModel)) {
    & $py scripts/export_spm_from_checkpoint.py `
        --checkpoint artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt `
        --out_model $spmModel
    if ($LASTEXITCODE -ne 0) {
        throw "failed to export spm model"
    }
}

$steps = 22000
$warmup = 300
$evalInterval = 250
$saveInterval = 1000
$maxRecords = 0

if ($Smoke) {
    $steps = 50
    $warmup = 20
    $evalInterval = 25
    $saveInterval = 25
    $maxRecords = 4096
}

$cmd = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/teacher_exact_short_latex_mix_v1_train.jsonl',
    '--output_dir', $outDir,
    '--max_records', "$maxRecords",
    '--steps', "$steps",
    '--batch_size', '1',
    '--grad_accum_steps', '16',
    '--seq_len', '384',
    '--lr', '1.5e-4',
    '--weight_decay', '0.1',
    '--warmup_steps', "$warmup",
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', "$evalInterval",
    '--save_interval', "$saveInterval",
    '--eval_batches', '16',
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
    '--turbo_min_seq_len', '256',
    '--turbo_seq_warmup_ratio', '0.35',
    '--turbo_max_frozen_layers', '4',
    '--turbo_depth_warmup_ratio', '0.25',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0',
    '--plain_sft'
)

if ($StartDetached) {
    $runnerPath = Join-Path $logDir ("exact_short_latex_runner_{0}.ps1" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $cmdLine = "& $(Quote-PSArg $py) " + (($cmd | ForEach-Object { Quote-PSArg $_ }) -join ' ')
    $runner = @(
        '$ErrorActionPreference = ''Stop'''
        "Set-Location $(Quote-PSArg $root)"
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
