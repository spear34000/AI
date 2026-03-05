param(
    [string]$RunTag = 'taskA',
    [int]$Steps = 22000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = 'C:\Program Files\Python311\python.exe'
$tokenDir = Join-Path $root 'artifacts_tokenizer_spm_from_stage2_v1'
$spmModel = Join-Path $tokenDir 'spm16k.model'
$outDir = Join-Path $root ("artifacts_exact_short_latex_50m_fromscratch_v1_{0}" -f $RunTag)
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir ("exact_short_latex_{0}.log" -f $RunTag)
$errFile = Join-Path $logDir ("exact_short_latex_{0}.err.log" -f $RunTag)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $tokenDir | Out-Null
if (-not (Test-Path $spmModel)) {
    & $py scripts/export_spm_from_checkpoint.py `
        --checkpoint artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt `
        --out_model $spmModel
    if ($LASTEXITCODE -ne 0) {
        throw "failed to export tokenizer"
    }
}

$args = @(
    '-u', 'scripts/train_slm.py',
    '--data_path', 'data/teacher_exact_short_latex_mix_v1_train.jsonl',
    '--output_dir', $outDir,
    '--max_records', '0',
    '--steps', "$Steps",
    '--batch_size', '1',
    '--grad_accum_steps', '16',
    '--seq_len', '384',
    '--lr', '1.5e-4',
    '--weight_decay', '0.1',
    '--warmup_steps', '300',
    '--grad_clip', '1.0',
    '--val_ratio', '0.02',
    '--eval_interval', '250',
    '--save_interval', '1000',
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

"[{0}] starting run {1}" -f (Get-Date -Format s), $RunTag | Out-File -FilePath $logFile -Encoding utf8 -Append
& $py @args 1>> $logFile 2>> $errFile
$code = $LASTEXITCODE
"[{0}] finished exit={1}" -f (Get-Date -Format s), $code | Out-File -FilePath $logFile -Encoding utf8 -Append
exit $code
