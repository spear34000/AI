param(
    [string]$BaseCheckpoint = "artifacts_final_continue_v2_corpus_mix/slm_best.pt",
    [string]$ExpandedCheckpoint = "artifacts_upgrade_126m_v1/slm_expanded_init.pt",
    [string]$TrainPath = "data/reasoning_blend_v4.jsonl",
    [string]$OutDir = "artifacts_upgrade_126m_v1_train",
    [int]$Steps = 1200,
    [int]$DModel = 768,
    [int]$NHeads = 12,
    [int]$NLayers = 16,
    [int]$SeqLen = 512,
    [switch]$ReExpand
)

$ErrorActionPreference = "Stop"

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

Push-Location $repoRoot
try {
    if (-not (Test-Path $BaseCheckpoint)) {
        throw "base checkpoint not found: $BaseCheckpoint"
    }
    if (-not (Test-Path $TrainPath)) {
        throw "train dataset not found: $TrainPath"
    }

    if ($ReExpand -or -not (Test-Path $ExpandedCheckpoint)) {
        Write-Output ("==> Expand checkpoint to d={0}, h={1}, l={2}, seq={3}" -f $DModel, $NHeads, $NLayers, $SeqLen)
        & $pythonExe scripts/expand_slm_checkpoint.py `
            --src $BaseCheckpoint `
            --dst $ExpandedCheckpoint `
            --new_d_model $DModel `
            --new_n_heads $NHeads `
            --new_n_layers $NLayers `
            --new_seq_len $SeqLen `
            --mlp_mult 4 `
            --dropout 0.08
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Output "==> Train upgraded model"
    & $pythonExe scripts/train_slm.py `
        --data_path $TrainPath `
        --output_dir $OutDir `
        --resume_from $ExpandedCheckpoint `
        --resume_weights_only `
        --reset_best_on_resume `
        --steps $Steps `
        --batch_size 1 `
        --grad_accum_steps 16 `
        --seq_len $SeqLen `
        --lr 1.2e-5 `
        --weight_decay 0.1 `
        --warmup_steps 80 `
        --grad_clip 1.0 `
        --val_ratio 0.02 `
        --ema_decay 0.0 `
        --consistency_lambda 0.0 `
        --consistency_temp 1.5 `
        --consistency_warmup 1000000 `
        --eval_interval 100 `
        --save_interval 200 `
        --eval_batches 8 `
        --num_workers 0 `
        --plain_sft `
        --activation_checkpointing `
        --skip_sample
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $bestPath = Join-Path $OutDir "slm_best.pt"
    if (-not (Test-Path $bestPath)) {
        $bestPath = Join-Path $OutDir "slm_last.pt"
    }

    Write-Output "==> Eval strict with clean tool cache"
    & $pythonExe scripts/eval_ko_quality5x.py `
        --candidate_checkpoint $bestPath `
        --device cuda `
        --tool_cache_path data/tool_knowledge_cache_v3_clean.jsonl `
        --out_json (Join-Path $OutDir "eval_ko_quality5x.json")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Output ("best_checkpoint={0}" -f $bestPath)
}
finally {
    Pop-Location
}
