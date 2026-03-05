param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,
    [Parameter(Mandatory = $true)]
    [string]$Stage2Dir,
    [Parameter(Mandatory = $true)]
    [string]$Stage3Dir,
    [string]$PythonExe = 'C:\Program Files\Python311\python.exe',
    [string]$TokenizerModel = 'artifacts_tokenizer_spm_from_direct_v3_v1/spm16k.model'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Run-Train {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgList
    )
    & $PythonExe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "train_slm.py failed with exit code $LASTEXITCODE"
    }
}

Write-Host ("[{0}] waiting for stage2 pid={1}" -f (Get-Date -Format s), $WaitPid)
try {
    Wait-Process -Id $WaitPid -ErrorAction Stop
} catch {
    Write-Host ("[{0}] stage2 pid wait ended with: {1}" -f (Get-Date -Format s), $_.Exception.Message)
}

$stage2Best = Join-Path $Stage2Dir 'slm_best.pt'
$stage2Last = Join-Path $Stage2Dir 'slm_last.pt'
if (Test-Path $stage2Best) {
    $resumeFrom = $stage2Best
} elseif (Test-Path $stage2Last) {
    $resumeFrom = $stage2Last
} else {
    throw "stage2 checkpoint not found in $Stage2Dir"
}

Write-Host ("[{0}] stage3 start from {1}" -f (Get-Date -Format s), $resumeFrom)
Run-Train -ArgList @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/logic_only_v6.jsonl',
    '--output_dir', $Stage3Dir,
    '--steps', '16800',
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
    '--tokenizer_model', $TokenizerModel,
    '--resume_from', $resumeFrom,
    '--resume_weights_only',
    '--reset_best_on_resume',
    '--activation_checkpointing',
    '--skip_sample',
    '--ema_decay', '0.0',
    '--consistency_lambda', '0.0',
    '--plain_sft'
)

Write-Host ("[{0}] stage3 finished" -f (Get-Date -Format s))
