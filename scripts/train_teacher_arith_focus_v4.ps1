param(
    [string]$RunTag = "runA",
    [string]$BaseCheckpoint = "artifacts_teacher_arith_pure_mix_fromscratch_v1_runB\\slm_best.pt",
    [switch]$StartDetached
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
$outDir = Join-Path $root ("artifacts_teacher_arith_focus_v4_{0}" -f $RunTag)
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir ("arith_focus_v4_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$errLogFile = [IO.Path]::ChangeExtension($logFile, '.err.log')

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

& $py scripts/build_teacher_arith_focus_v4.py
if ($LASTEXITCODE -ne 0) {
    throw "dataset build failed"
}

$cmd = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/teacher_arith_focus_v4_train.jsonl',
    '--output_dir', $outDir,
    '--steps', '23800',
    '--batch_size', '1',
    '--grad_accum_steps', '16',
    '--seq_len', '384',
    '--lr', '3.0e-5',
    '--weight_decay', '0.05',
    '--warmup_steps', '80',
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', '100',
    '--save_interval', '400',
    '--eval_batches', '16',
    '--num_workers', '0',
    '--resume_from', $BaseCheckpoint,
    '--resume_weights_only',
    '--plain_sft',
    '--activation_checkpointing',
    '--skip_sample',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0'
)

if ($StartDetached) {
    $runnerPath = Join-Path $logDir ("arith_focus_v4_runner_{0}.ps1" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
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
