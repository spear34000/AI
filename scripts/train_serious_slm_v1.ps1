param(
    [switch]$RebuildCorpus,
    [switch]$RetrainTokenizer,
    [ValidateSet("pilot", "full")]
    [string]$Preset = "pilot",
    [string]$CorpusPath = "data/serious_slm_corpus_v1.jsonl",
    [string]$CorpusManifest = "data/serious_slm_corpus_v1.manifest.json",
    [string]$TokenizerDir = "artifacts_tokenizer_spm_serious_v1",
    [string]$TokenizerPrefix = "serious_ko_spm16k_v1",
    [int]$TokenizerVocabSize = 16000,
    [string]$StagesDir = "data/stages_serious_slm_v1",
    [string]$RunDir = "artifacts_serious_slm_v1"
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

$repoRoot = Split-Path $PSScriptRoot -Parent
$pySpm = Resolve-PythonExe -repoRoot $repoRoot -RequireSentencePiece $true
$pyTorch = Resolve-PythonExe -repoRoot $repoRoot -RequireTorch $true

$tokenizerModel = Join-Path $TokenizerDir ($TokenizerPrefix + ".model")

Push-Location $repoRoot
try {
    if ($RebuildCorpus -or -not (Test-Path $CorpusPath)) {
        Write-Output "==> Build serious curated corpus"
        & $pyTorch scripts/build_serious_slm_corpus_v1.py --out_jsonl $CorpusPath --manifest $CorpusManifest --shuffle
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($RetrainTokenizer -or -not (Test-Path $tokenizerModel)) {
        Write-Output "==> Train serious SPM tokenizer"
        & $pySpm scripts/train_spm_tokenizer.py `
            --data_paths $CorpusPath `
            --out_dir $TokenizerDir `
            --model_prefix $TokenizerPrefix `
            --vocab_size $TokenizerVocabSize `
            --character_coverage 0.9995 `
            --model_type unigram `
            --normalization nmt_nfkc `
            --max_line_chars 1600
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Output ("==> Train serious SLM ({0})" -f $Preset)
    & $PSScriptRoot\train_rtx4060_turbo.ps1 `
        -SourceJsonl $CorpusPath `
        -StagesDir $StagesDir `
        -RunDir $RunDir `
        -LicensePolicy allow_missing `
        -TokenizerType spm `
        -TokenizerModel $tokenizerModel `
        -Preset $Preset `
        -ExportOnDeviceBest $false
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
