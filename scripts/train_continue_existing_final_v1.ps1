param(
    [string]$ResumeFrom = "artifacts_serious_slm_v2_defsboost/slm_best.pt",
    [string]$TokenizerModel = "artifacts_tokenizer_spm_serious_v1/serious_ko_spm16k_v1.model",
    [string]$CorpusPath = "data/final_datasets_corpus_v1.jsonl",
    [string]$DefsBoostPath = "data/final_datasets_defs_boost_v1.jsonl",
    [string]$ToolCachePath = "data/tool_knowledge_cache_final_v1.jsonl",
    [string]$Phase1OutDir = "artifacts_final_continue_v1_corpus",
    [string]$Phase2OutDir = "artifacts_final_continue_v1_defsboost",
    [int]$Phase1Steps = 2200,
    [int]$Phase2Steps = 1200
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath, [bool]$RequireTorch = $false) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        if ($RequireTorch) {
            & $exePath -c "import torch" *> $null
            if ($LASTEXITCODE -ne 0) {
                return $false
            }
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

function Resolve-BestCheckpoint([string]$dirPath) {
    $best = Join-Path $dirPath "slm_best.pt"
    if (Test-Path $best) { return $best }
    $last = Join-Path $dirPath "slm_last.pt"
    if (Test-Path $last) { return $last }
    throw "checkpoint not found in $dirPath"
}

function Resolve-ExistingCheckpoint([string]$dirPath) {
    $best = Join-Path $dirPath "slm_best.pt"
    if (Test-Path $best) { return $best }
    $last = Join-Path $dirPath "slm_last.pt"
    if (Test-Path $last) { return $last }
    return ""
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

Push-Location $repoRoot
try {
    if (-not (Test-Path $ResumeFrom)) {
        throw "resume checkpoint not found: $ResumeFrom"
    }
    if (-not (Test-Path $TokenizerModel)) {
        throw "tokenizer model not found: $TokenizerModel"
    }
    if (-not (Test-Path $CorpusPath)) {
        throw "corpus not found: $CorpusPath"
    }
    if (-not (Test-Path $DefsBoostPath)) {
        throw "defs boost not found: $DefsBoostPath"
    }

    $phase1Resume = Resolve-ExistingCheckpoint -dirPath $Phase1OutDir
    if (-not $phase1Resume) {
        $phase1Resume = $ResumeFrom
    }

    Write-Output ("==> Phase 1: continue on final corpus (resume={0})" -f $phase1Resume)
    & $pythonExe scripts/train_slm.py `
        --data_path $CorpusPath `
        --output_dir $Phase1OutDir `
        --resume_from $phase1Resume `
        --resume_weights_only `
        --reset_best_on_resume `
        --tokenizer_type spm `
        --tokenizer_model $TokenizerModel `
        --steps $Phase1Steps `
        --batch_size 1 `
        --grad_accum_steps 16 `
        --seq_len 512 `
        --lr 3e-5 `
        --weight_decay 0.1 `
        --warmup_steps 80 `
        --grad_clip 1.0 `
        --val_ratio 0.02 `
        --ema_decay 0.0 `
        --consistency_lambda 0.01 `
        --consistency_temp 1.5 `
        --consistency_warmup 400 `
        --eval_interval 100 `
        --save_interval 200 `
        --eval_batches 8 `
        --num_workers 0 `
        --activation_checkpointing `
        --skip_sample
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $phase1Best = Resolve-BestCheckpoint -dirPath $Phase1OutDir
    $phase2Resume = Resolve-ExistingCheckpoint -dirPath $Phase2OutDir
    if (-not $phase2Resume) {
        $phase2Resume = $phase1Best
    }

    Write-Output ("==> Phase 2: continue on final defs boost (resume={0})" -f $phase2Resume)
    & $pythonExe scripts/train_slm.py `
        --data_path $DefsBoostPath `
        --output_dir $Phase2OutDir `
        --resume_from $phase2Resume `
        --resume_weights_only `
        --reset_best_on_resume `
        --tokenizer_type spm `
        --tokenizer_model $TokenizerModel `
        --steps $Phase2Steps `
        --batch_size 1 `
        --grad_accum_steps 16 `
        --seq_len 512 `
        --lr 2e-5 `
        --weight_decay 0.1 `
        --warmup_steps 50 `
        --grad_clip 1.0 `
        --val_ratio 0.02 `
        --ema_decay 0.0 `
        --consistency_lambda 0.0 `
        --consistency_temp 1.5 `
        --consistency_warmup 1000000 `
        --eval_interval 50 `
        --save_interval 100 `
        --eval_batches 8 `
        --num_workers 0 `
        --plain_sft `
        --activation_checkpointing `
        --skip_sample
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $phase2Best = Resolve-BestCheckpoint -dirPath $Phase2OutDir

    Write-Output "==> Eval strict with final tool cache"
    & $pythonExe scripts/eval_ko_quality5x.py `
        --candidate_checkpoint $phase2Best `
        --device cuda `
        --tool_cache_path $ToolCachePath `
        --out_json (Join-Path $Phase2OutDir "eval_ko_quality5x_final.json")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Output ("phase1_best={0}" -f $phase1Best)
    Write-Output ("phase2_best={0}" -f $phase2Best)
    Write-Output ("tool_cache={0}" -f $ToolCachePath)
}
finally {
    Pop-Location
}
