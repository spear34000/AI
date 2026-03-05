param(
    [string]$ResumeFrom = "artifacts_final_continue_v2_corpus_mix/slm_best.pt",
    [string]$TrainPath = "data/reasoning_blend_v3.jsonl",
    [string]$ManifestPath = "data/reasoning_blend_v3.manifest.json",
    [string]$OutDir = "artifacts_reasoning_blend_v3",
    [int]$Steps = 3600,
    [switch]$Rebuild
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
    if (-not (Test-Path $ResumeFrom)) {
        throw "resume checkpoint not found: $ResumeFrom"
    }

    if ($Rebuild -or -not (Test-Path $TrainPath)) {
        Write-Output "==> Build reasoning blend v3"
        & $pythonExe scripts/build_reasoning_blend_v3.py --out_jsonl $TrainPath --manifest $ManifestPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (-not (Test-Path $TrainPath)) {
        throw "train dataset not found: $TrainPath"
    }

    Write-Output ("==> Train reasoning blend v3 (resume={0})" -f $ResumeFrom)
    & $pythonExe scripts/train_slm.py `
        --data_path $TrainPath `
        --output_dir $OutDir `
        --resume_from $ResumeFrom `
        --resume_weights_only `
        --reset_best_on_resume `
        --steps $Steps `
        --batch_size 1 `
        --grad_accum_steps 16 `
        --seq_len 512 `
        --lr 2e-5 `
        --weight_decay 0.1 `
        --warmup_steps 60 `
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

    Write-Output "==> Eval strict"
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
