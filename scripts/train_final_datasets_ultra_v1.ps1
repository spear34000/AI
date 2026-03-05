param(
    [switch]$RebuildCorpus,
    [switch]$RetrainTokenizer,
    [switch]$RebuildDefsBoost,
    [switch]$RebuildToolCache,
    [switch]$BuildOnly,
    [switch]$IncludePersona,
    [ValidateSet("pilot", "full")]
    [string]$Preset = "full",
    [string]$FinalDatasetsDir = "data/final_datasets",
    [string]$CorpusPath = "data/final_datasets_corpus_v1.jsonl",
    [string]$CorpusManifest = "data/final_datasets_corpus_v1.manifest.json",
    [string]$DefsBoostPath = "data/final_datasets_defs_boost_v1.jsonl",
    [string]$DefsBoostManifest = "data/final_datasets_defs_boost_v1.manifest.json",
    [string]$ToolCachePath = "data/tool_knowledge_cache_final_v1.jsonl",
    [string]$TokenizerDir = "artifacts_tokenizer_spm_final_v1",
    [string]$TokenizerPrefix = "final_ko_spm16k_v1",
    [string]$StagesDir = "data/stages_final_datasets_ultra_v1",
    [string]$BaseRunDir = "artifacts_final_datasets_ultra_v1",
    [string]$DefsRunDir = "artifacts_final_datasets_ultra_v1_defsboost",
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

function Add-SourceSpecArgs([System.Collections.ArrayList]$argList, [string[]]$specs) {
    foreach ($spec in $specs) {
        [void]$argList.Add("--source_spec")
        [void]$argList.Add($spec)
    }
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pyTorch = Resolve-PythonExe -repoRoot $repoRoot -RequireTorch $true
$rootDir = Join-Path $repoRoot $FinalDatasetsDir
$planPath = Join-Path $repoRoot "data/final_datasets_plan_v1.json"
if (-not (Test-Path $rootDir)) {
    throw "final datasets dir not found: $rootDir"
}

$plannerArgs = @(
    "scripts/plan_final_datasets_v1.py",
    "--root", $FinalDatasetsDir,
    "--out_json", $planPath
)
if ($IncludePersona) {
    $plannerArgs += @("--include_persona")
}
$null = & $pyTorch @plannerArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$plan = Get-Content -Path $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($null -eq $plan) {
    throw "failed to parse final_datasets plan"
}
$baseSpecs = @($plan.base_specs)
$defsSpecs = @($plan.defs_specs)
if ($baseSpecs.Count -eq 0) {
    throw "base_specs is empty"
}
if ($defsSpecs.Count -eq 0) {
    throw "defs_specs is empty"
}

Push-Location $repoRoot
try {
    if ($RebuildCorpus -or -not (Test-Path $CorpusPath)) {
        Write-Output "==> Build corpus from data/final_datasets"
        $argsList = [System.Collections.ArrayList]@(
            "scripts/build_serious_slm_corpus_v1.py",
            "--out_jsonl", $CorpusPath,
            "--manifest", $CorpusManifest,
            "--shuffle"
        )
        Add-SourceSpecArgs -argList $argsList -specs $baseSpecs
        & $pyTorch @argsList
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($RebuildDefsBoost -or -not (Test-Path $DefsBoostPath)) {
        Write-Output "==> Build defs boost from data/final_datasets"
        $argsList = [System.Collections.ArrayList]@(
            "scripts/build_serious_slm_corpus_v1.py",
            "--out_jsonl", $DefsBoostPath,
            "--manifest", $DefsBoostManifest,
            "--shuffle"
        )
        Add-SourceSpecArgs -argList $argsList -specs $defsSpecs
        & $pyTorch @argsList
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($RebuildToolCache -or -not (Test-Path $ToolCachePath)) {
        Write-Output "==> Build tool cache from final_datasets"
        $cacheInputs = @($DefsBoostPath)
        foreach ($spec in $defsSpecs) {
            $src = [string]$spec
            $cut = $src.LastIndexOf(":")
            if ($cut -gt 1) {
                $src = $src.Substring(0, $cut)
            }
            if ($cacheInputs -notcontains $src) {
                $cacheInputs += $src
            }
        }
        & $pyTorch scripts/build_local_def_tool_cache_v1.py --inputs $cacheInputs --out_jsonl $ToolCachePath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($BuildOnly) {
        Write-Output "==> Build only complete"
        Write-Output ("corpus={0}" -f $CorpusPath)
        Write-Output ("defs_boost={0}" -f $DefsBoostPath)
        Write-Output ("tool_cache={0}" -f $ToolCachePath)
        exit 0
    }

    & $PSScriptRoot\train_ultra_slm_v1.ps1 `
        -Preset $Preset `
        -CorpusPath $CorpusPath `
        -CorpusManifest $CorpusManifest `
        -TokenizerDir $TokenizerDir `
        -TokenizerPrefix $TokenizerPrefix `
        -StagesDir $StagesDir `
        -BaseRunDir $BaseRunDir `
        -DefsBoostPath $DefsBoostPath `
        -DefsBoostManifest $DefsBoostManifest `
        -DefsRunDir $DefsRunDir `
        -ToolCachePath $ToolCachePath `
        -DefsBoostSteps $DefsBoostSteps `
        -RebuildCorpus:$false `
        -RetrainTokenizer:$RetrainTokenizer `
        -RebuildDefsBoost:$false `
        -RebuildToolCache:$false
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

