param(
    [switch]$RebuildCorpus,
    [switch]$RetrainTokenizer,
    [switch]$RebuildDefsBoost,
    [switch]$RebuildToolCache,
    [ValidateSet("pilot", "full")]
    [string]$Preset = "full",
    [string]$CorpusPath = "data/serious_slm_corpus_v1.jsonl",
    [string]$CorpusManifest = "data/serious_slm_corpus_v1.manifest.json",
    [string]$TokenizerDir = "artifacts_tokenizer_spm_serious_v1",
    [string]$TokenizerPrefix = "serious_ko_spm16k_v1",
    [string]$StagesDir = "data/stages_serious_slm_ultra_v1",
    [string]$BaseRunDir = "artifacts_serious_slm_ultra_v1",
    [string]$DefsBoostPath = "data/serious_defs_boost_v1.jsonl",
    [string]$DefsBoostManifest = "data/serious_defs_boost_v1.manifest.json",
    [string]$DefsRunDir = "artifacts_serious_slm_ultra_v1_defsboost",
    [string]$ToolCachePath = "data/tool_knowledge_cache_v3_clean.jsonl",
    [int]$DefsBoostSteps = 1600
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath, [bool]$RequireTorch = $false, [bool]$RequireSentencePiece = $false) {
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
        if ($RequireSentencePiece) {
            & $exePath -c "import sentencepiece" *> $null
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

function Resolve-PythonExe([string]$repoRoot, [bool]$RequireTorch = $false, [bool]$RequireSentencePiece = $false) {
    $localVenv = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $candidates = @(
        $localVenv,
        "C:\Program Files\Python311\python.exe"
    )

    foreach ($c in $candidates) {
        if (Test-PythonExe -exePath $c -RequireTorch $RequireTorch -RequireSentencePiece $RequireSentencePiece) {
            return $c
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonExe -exePath $cmd.Source -RequireTorch $RequireTorch -RequireSentencePiece $RequireSentencePiece)) {
        return $cmd.Source
    }

    throw "No usable python executable found."
}

function Resolve-BestCheckpoint([string]$stageDir) {
    $best = Join-Path $stageDir "slm_best.pt"
    if (Test-Path $best) { return $best }
    $last = Join-Path $stageDir "slm_last.pt"
    if (Test-Path $last) { return $last }
    throw "checkpoint not found in $stageDir"
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pyTorch = Resolve-PythonExe -repoRoot $repoRoot -RequireTorch $true
$pySpm = Resolve-PythonExe -repoRoot $repoRoot -RequireSentencePiece $true
$tokenizerModel = Join-Path $TokenizerDir ($TokenizerPrefix + ".model")

Push-Location $repoRoot
try {
    if ($RebuildDefsBoost -or -not (Test-Path $DefsBoostPath)) {
        Write-Output "==> Rebuild definition boost corpus"
        & $pyTorch scripts/build_serious_slm_corpus_v1.py `
            --out_jsonl $DefsBoostPath `
            --manifest $DefsBoostManifest `
            --shuffle `
            --source_spec data/clean_mix_v3.jsonl:20000 `
            --source_spec data/deintro_focus_v1.jsonl:20000 `
            --source_spec data/term_focus_clean_v1.jsonl:16000 `
            --source_spec data/term_anchor_patch_v2.jsonl:14000 `
            --source_spec data/ko_def_grounding_patch_v1.jsonl:15000 `
            --source_spec data/ko_targeted_shortanswer_v3.jsonl:20000
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Output ("==> Base serious SLM ({0})" -f $Preset)
    & $PSScriptRoot\train_serious_slm_v1.ps1 `
        -RebuildCorpus:$RebuildCorpus `
        -RetrainTokenizer:$RetrainTokenizer `
        -Preset $Preset `
        -CorpusPath $CorpusPath `
        -CorpusManifest $CorpusManifest `
        -TokenizerDir $TokenizerDir `
        -TokenizerPrefix $TokenizerPrefix `
        -StagesDir $StagesDir `
        -RunDir $BaseRunDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not (Test-Path $tokenizerModel)) {
        throw "tokenizer model not found: $tokenizerModel"
    }

    $baseCheckpoint = Resolve-BestCheckpoint -stageDir (Join-Path $BaseRunDir "stage4_instruction")
    Write-Output "==> Definition boost fine-tune"
    & $pyTorch scripts/train_slm.py `
        --data_path $DefsBoostPath `
        --output_dir $DefsRunDir `
        --resume_from $baseCheckpoint `
        --resume_weights_only `
        --reset_best_on_resume `
        --tokenizer_type spm `
        --tokenizer_model $tokenizerModel `
        --steps $DefsBoostSteps `
        --batch_size 1 `
        --grad_accum_steps 16 `
        --seq_len 512 `
        --lr 5e-5 `
        --weight_decay 0.1 `
        --warmup_steps 40 `
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

    if ($RebuildToolCache -or -not (Test-Path $ToolCachePath)) {
        Write-Output "==> Build local definition tool cache"
        & $pyTorch scripts/build_local_def_tool_cache_v1.py `
            --inputs `
                $DefsBoostPath `
                data/deintro_focus_v1.jsonl `
                data/term_focus_clean_v1.jsonl `
                data/term_anchor_patch_v2.jsonl `
                data/ko_def_grounding_patch_v1.jsonl `
                data/ko_targeted_shortanswer_v3.jsonl `
                data/clean_mix_v3.jsonl `
            --out_jsonl $ToolCachePath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Output "==> Final checkpoints"
    Write-Output ("base_stage4={0}" -f $baseCheckpoint)
    Write-Output ("defsboost_best={0}" -f (Resolve-BestCheckpoint -stageDir $DefsRunDir))
    Write-Output ("tool_cache={0}" -f $ToolCachePath)
}
finally {
    Pop-Location
}
