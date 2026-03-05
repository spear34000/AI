param(
    [string]$SourceJsonl = "data/slm_mit_unified_v4.jsonl",
    [string]$SpecPath = "data/ccl_specbook_v1.jsonl",
    [string]$SpecManifest = "data/ccl_specbook_v1.manifest.json",
    [string]$BaseCheckpoint = "artifacts_ondevice_best/slm_ondevice_fp16.pt",
    [string]$OutputDir = "artifacts_ccl2_compile",
    [ValidateSet("pilot", "full")]
    [string]$Preset = "full",
    [int]$Seed = 42,
    [switch]$RebuildSpec
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        return ($LASTEXITCODE -eq 0)
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
        if (Test-PythonExe -exePath $c) {
            return $c
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonExe -exePath $cmd.Source)) {
        return $cmd.Source
    }

    throw "No usable python executable found."
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

$maxSpecsBuild = 2400
$trainArgs = @{
    BaseCheckpoint = $BaseCheckpoint
    SpecPath = $SpecPath
    DataPath = $SourceJsonl
    OutputDir = $OutputDir
    Steps = 80
    MaxSpecs = 1800
    MineSeeds = 32
    MutationsPerSeed = 4
    CounterexamplesPerStep = 8
    CompileBatchSize = 6
    VerifyInterval = 4
    VerifySpecs = 300
    TargetHardPass = 0.95
    PatchTopLayers = 4
    PatchTargets = "qkv,proj,mlp_in,mlp_out"
    LoraRank = 8
    LoraAlpha = 16
    LoraDropout = 0.0
    CgIters = 10
    CgTol = 1e-4
    Damping = 0.08
    StepScale = 0.8
    MaxUpdateNorm = 0.08
    LedgerMaxBasis = 24
    LedgerRefreshInterval = 3
    LedgerRefreshSamples = 6
    NoveltyThreshold = 0.92
    SaveInterval = 8
    Seed = $Seed
    SeqLen = 0
    Device = "cuda"
    Mutators = "identity,boundary,constraint_clash,reorder,ambiguity,whitespace,length_cap"
    MaxNewTokens = 96
}

if ($Preset -eq "pilot") {
    $maxSpecsBuild = 900
    $trainArgs["Steps"] = 24
    $trainArgs["MaxSpecs"] = 640
    $trainArgs["MineSeeds"] = 16
    $trainArgs["CounterexamplesPerStep"] = 6
    $trainArgs["CompileBatchSize"] = 4
    $trainArgs["VerifySpecs"] = 140
    $trainArgs["PatchTopLayers"] = 2
    $trainArgs["LoraRank"] = 4
    $trainArgs["LoraAlpha"] = 8
    $trainArgs["CgIters"] = 6
    $trainArgs["SaveInterval"] = 4
}

Push-Location $repoRoot
try {
    if ($RebuildSpec.IsPresent -or -not (Test-Path $SpecPath)) {
        Write-Output ("==> Build CCL spec book ({0} specs)" -f $maxSpecsBuild)
        & $pythonExe "scripts/build_ccl_specbook.py" `
            --source_jsonl $SourceJsonl `
            --output_jsonl $SpecPath `
            --manifest_json $SpecManifest `
            --max_specs $maxSpecsBuild `
            --seed $Seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Output "==> Reusing existing CCL spec book"
    }

    Write-Output ("==> Run CCL compile preset: {0}" -f $Preset)
    & "$PSScriptRoot/train_ccl_compile.ps1" @trainArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
