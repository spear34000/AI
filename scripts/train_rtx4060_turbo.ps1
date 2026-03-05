param(
    [string]$SourceJsonl = "data/slm_mit_unified_v4.jsonl",
    [string]$StagesDir = "data/stages_4060_turbo",
    [string]$RunDir = "artifacts_rtx4060_turbo",
    [ValidateSet("mit_only", "allow_missing", "all")]
    [string]$LicensePolicy = "mit_only",
    [ValidateSet("byte", "spm")]
    [string]$TokenizerType = "byte",
    [string]$TokenizerModel = "",
    [ValidateSet("pilot", "full")]
    [string]$Preset = "full",
    [int]$Seed = 42,
    [bool]$ExportOnDeviceBest = $true
)

$ErrorActionPreference = "Stop"

$trainArgs = @{
    SourceJsonl = $SourceJsonl
    StagesDir = $StagesDir
    RunDir = $RunDir
    LicensePolicy = $LicensePolicy
    TokenizerType = $TokenizerType
    Seed = $Seed
    SeqLen = 512
    BatchSize = 1
    GradAccumSteps = 16
    DModel = 640
    NHeads = 10
    NLayers = 14
    MlpMult = 4
    Dropout = 0.1
    WeightDecay = 0.1
    GradClip = 1.0
    ValRatio = 0.02
    NumWorkers = 0
    EvalBatches = 8
    EvalInterval = 200
    SaveInterval = 800
    ActivationCheckpointing = $true
    SkipBuildIfReady = $true
    SaveStepCheckpoints = $false
    SkipSample = $true
    SaveBestCheckpoint = $true
    FastMode = $false
    TurboMode = $true
    TurboMinSeqLen = 96
    TurboSeqWarmupRatio = 0.62
    TurboMaxFrozenLayers = 6
    TurboDepthWarmupRatio = 0.40
    TurboFreezeEmbeddings = $true
}

if ($TokenizerType -eq "spm") {
    $trainArgs["TokenizerModel"] = $TokenizerModel
}

if ($Preset -eq "pilot") {
    # Fast sanity run for pipeline validation.
    $trainArgs["MaxTotalTokens"] = 8000000
    $trainArgs["Stage1Tokens"] = 3500000
    $trainArgs["Stage2Tokens"] = 1800000
    $trainArgs["Stage3Tokens"] = 1800000
    $trainArgs["Stage4Tokens"] = 900000
    $trainArgs["FastMode"] = $true
    $trainArgs["EvalInterval"] = 100
    $trainArgs["SaveInterval"] = 400
} else {
    # Main run tuned for single RTX 4060.
    $trainArgs["MaxTotalTokens"] = 42000000
    $trainArgs["Stage1Tokens"] = 18000000
    $trainArgs["Stage2Tokens"] = 9000000
    $trainArgs["Stage3Tokens"] = 11000000
    $trainArgs["Stage4Tokens"] = 4000000
}

Write-Output ("==> RTX4060 Turbo preset: {0}" -f $Preset)
& "$PSScriptRoot/train_50m_4stage.ps1" @trainArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($ExportOnDeviceBest) {
    Write-Output "==> Export best on-device checkpoint"
    & "$PSScriptRoot/build_best_ondevice.ps1" -OutputDir (Join-Path $RunDir "ondevice_best") -DType "fp16" -ProbeTopN 3
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
