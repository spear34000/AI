param(
    [switch]$Smoke,
    [switch]$StartDetached,
    [ValidateSet('50m_safe','79m_edge')]
    [string]$Preset = '50m_safe',
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

switch ($Preset) {
    '50m_safe' {
        $dModel = 576
        $nHeads = 9
        $nLayers = 12
        $mlpMult = 4
    }
    '79m_edge' {
        $dModel = 640
        $nHeads = 10
        $nLayers = 14
        $mlpMult = 4
    }
}

$py = Get-TrainPython
$tokenDir = Join-Path $root 'artifacts_tokenizer_spm_from_direct_v3_v1'
$spmModel = Join-Path $tokenDir 'spm16k.model'
$baseName = "artifacts_bigslm_${Preset}_fromscratch_v1"
if ($RunTag) {
    $baseName = "${baseName}_$RunTag"
}
$baseOut = Join-Path $root $baseName
$stage1Out = "${baseOut}_stage1"
$stage2Out = "${baseOut}_stage2"
$stage3Out = "${baseOut}_stage3"
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir ("train_bigslm_{0}_{1}.log" -f $Preset, (Get-Date -Format 'yyyyMMdd_HHmmss'))

New-Item -ItemType Directory -Force -Path $tokenDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $spmModel)) {
    & $py scripts/export_spm_from_checkpoint.py `
        --checkpoint artifacts_ko_nometa_direct_v3/slm_best.pt `
        --out_model $spmModel
}

$stage1Steps = 18000
$stage2Steps = 4800
$stage3Steps = 1800
$maxRecords = 0
$stage1Warmup = 200
$stage1EvalInterval = 200
$stage1SaveInterval = 400

if ($Smoke) {
    $stage1Steps = 20
    $stage2Steps = 0
    $stage3Steps = 0
    $maxRecords = 4096
    $stage1Warmup = 20
    $stage1EvalInterval = 20
    $stage1SaveInterval = 20
}

$cmd1 = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/final_datasets_corpus_v1.jsonl',
    '--output_dir', $stage1Out,
    '--max_records', "$maxRecords",
    '--steps', "$stage1Steps",
    '--batch_size', '1',
    '--grad_accum_steps', '16',
    '--seq_len', '384',
    '--lr', '1.5e-4',
    '--weight_decay', '0.1',
    '--warmup_steps', "$stage1Warmup",
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', "$stage1EvalInterval",
    '--save_interval', "$stage1SaveInterval",
    '--eval_batches', '8',
    '--num_workers', '0',
    '--d_model', "$dModel",
    '--n_heads', "$nHeads",
    '--n_layers', "$nLayers",
    '--mlp_mult', "$mlpMult",
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

$cmd2 = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/repair_ko_direct_v3.jsonl',
    '--output_dir', $stage2Out,
    '--steps', "$stage2Steps",
    '--batch_size', '1',
    '--grad_accum_steps', '16',
    '--seq_len', '384',
    '--lr', '3.0e-5',
    '--weight_decay', '0.05',
    '--warmup_steps', '80',
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', '100',
    '--save_interval', '200',
    '--eval_batches', '8',
    '--num_workers', '0',
    '--tokenizer_type', 'spm',
    '--tokenizer_model', $spmModel,
    '--resume_from', (Join-Path $stage1Out 'slm_best.pt'),
    '--resume_weights_only',
    '--reset_best_on_resume',
    '--activation_checkpointing',
    '--skip_sample',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0',
    '--plain_sft'
)

$cmd3 = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/logic_only_v6.jsonl',
    '--output_dir', $stage3Out,
    '--steps', "$stage3Steps",
    '--batch_size', '1',
    '--grad_accum_steps', '16',
    '--seq_len', '384',
    '--lr', '8.0e-6',
    '--weight_decay', '0.02',
    '--warmup_steps', '50',
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', '100',
    '--save_interval', '200',
    '--eval_batches', '8',
    '--num_workers', '0',
    '--tokenizer_type', 'spm',
    '--tokenizer_model', $spmModel,
    '--resume_from', (Join-Path $stage2Out 'slm_best.pt'),
    '--resume_weights_only',
    '--reset_best_on_resume',
    '--activation_checkpointing',
    '--skip_sample',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0',
    '--plain_sft'
)

function Run-Train([string]$PythonExe, [string[]]$Args) {
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "train_slm.py failed with exit code $LASTEXITCODE"
    }
}

function Quote-PSArg([string]$Value) {
    if ($null -eq $Value) { return "''" }
    return "'" + ($Value -replace "'", "''") + "'"
}

if ($StartDetached) {
    $runnerPath = Join-Path $logDir ("train_bigslm_runner_{0}_{1}.ps1" -f $Preset, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $errLogFile = [IO.Path]::ChangeExtension($logFile, '.err.log')
    $cmd1Line = "& $(Quote-PSArg $py) " + (($cmd1 | ForEach-Object { Quote-PSArg $_ }) -join ' ')
    $cmd2Line = "& $(Quote-PSArg $py) " + (($cmd2 | ForEach-Object { Quote-PSArg $_ }) -join ' ')
    $cmd3Line = "& $(Quote-PSArg $py) " + (($cmd3 | ForEach-Object { Quote-PSArg $_ }) -join ' ')
    $runner = @(
        '$ErrorActionPreference = ''Stop'''
        "Set-Location $(Quote-PSArg $root)"
        $cmd1Line
    )
    if (-not $Smoke) {
        $runner += "if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }"
        $runner += $cmd2Line
        $runner += "if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }"
        $runner += $cmd3Line
    }
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
        stage1 = $stage1Out
        stage2 = $stage2Out
        stage3 = $stage3Out
        preset = $Preset
    } | ConvertTo-Json -Compress
    exit 0
}

Run-Train $py $cmd1
if (-not $Smoke) {
    Run-Train $py $cmd2
    Run-Train $py $cmd3
}
