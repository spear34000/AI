param(
    [string]$SourceJsonl = "data/slm_mit_unified_v4.jsonl",
    [string]$StagesDir = "data/stages_50m",
    [string]$RunDir = "artifacts_50m_4stage",
    [ValidateSet("mit_only", "allow_missing", "all")]
    [string]$LicensePolicy = "mit_only",
    [ValidateSet("byte", "spm")]
    [string]$TokenizerType = "byte",
    [string]$TokenizerModel = "",
    [int]$Seed = 42,
    [int]$MaxTotalTokens = 50000000,
    [int]$Stage1Tokens = 22000000,
    [int]$Stage2Tokens = 10000000,
    [int]$Stage3Tokens = 14000000,
    [int]$Stage4Tokens = 4000000,
    [int]$SeqLen = 512,
    [int]$BatchSize = 1,
    [int]$GradAccumSteps = 16,
    [int]$DModel = 1024,
    [int]$NHeads = 16,
    [int]$NLayers = 24,
    [int]$MlpMult = 4,
    [double]$Dropout = 0.1,
    [double]$WeightDecay = 0.1,
    [double]$GradClip = 1.0,
    [double]$ValRatio = 0.02,
    [int]$NumWorkers = 0,
    [int]$EvalBatches = 8,
    [int]$EvalInterval = 200,
    [int]$SaveInterval = 800,
    [bool]$ActivationCheckpointing = $false,
    [bool]$SkipBuildIfReady = $true,
    [bool]$SaveStepCheckpoints = $false,
    [bool]$SkipSample = $true,
    [bool]$SaveBestCheckpoint = $false,
    [bool]$FastMode = $true,
    [bool]$TurboMode = $false,
    [int]$TurboMinSeqLen = 128,
    [double]$TurboSeqWarmupRatio = 0.55,
    [int]$TurboMaxFrozenLayers = 4,
    [double]$TurboDepthWarmupRatio = 0.35,
    [bool]$TurboFreezeEmbeddings = $false
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

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

function Get-StepsFromTokens([int]$tokens, [int]$seqLen, [int]$batchSize, [int]$gradAccum) {
    $perStep = [int]($seqLen * $batchSize * $gradAccum)
    if ($perStep -le 0) { throw "invalid effective tokens per step: $perStep" }
    return [int][Math]::Ceiling($tokens / $perStep)
}

function Resolve-StageCheckpoint([string]$stageDir) {
    $last = Join-Path $stageDir "slm_last.pt"
    if (Test-Path $last) { return $last }
    $best = Join-Path $stageDir "slm_best.pt"
    if (Test-Path $best) { return $best }
    throw "checkpoint not found in $stageDir"
}

function Resolve-ResumeCheckpoint([string]$stageDir) {
    $last = Join-Path $stageDir "slm_last.pt"
    if (Test-Path $last) { return $last }
    $best = Join-Path $stageDir "slm_best.pt"
    if (Test-Path $best) { return $best }
    return ""
}

function Get-CheckpointStep([string]$ckptPath) {
    if (-not $ckptPath -or -not (Test-Path $ckptPath)) {
        return -1
    }
    $stepRaw = & $pythonExe -c "import sys, torch; ckpt=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(int(ckpt.get('step', 0)))" "$ckptPath"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to read checkpoint step: $ckptPath"
    }
    return [int]$stepRaw
}

function Test-StageFilesReady([string]$stageDir) {
    $need = @(
        (Join-Path $stageDir "stage_manifest.json"),
        (Join-Path $stageDir "stage1_base_lm.jsonl"),
        (Join-Path $stageDir "stage2_curriculum.jsonl"),
        (Join-Path $stageDir "stage3_distill.jsonl"),
        (Join-Path $stageDir "stage4_instruction.jsonl")
    )
    foreach ($p in $need) {
        if (-not (Test-Path $p)) { return $false }
    }
    return $true
}

function Test-StageManifestMatches([string]$stageDir) {
    $manifestPath = Join-Path $stageDir "stage_manifest.json"
    if (-not (Test-Path $manifestPath)) { return $false }
    try {
        $m = Get-Content $manifestPath -Raw | ConvertFrom-Json
    } catch {
        return $false
    }
    if ($null -eq $m) { return $false }
    if ([int]$m.max_total_tokens -ne [int]$MaxTotalTokens) { return $false }
    if ([int]$m.token_budget_sum -ne ([int]$Stage1Tokens + [int]$Stage2Tokens + [int]$Stage3Tokens + [int]$Stage4Tokens)) { return $false }
    if ($null -eq $m.stages) { return $false }
    if ([int]$m.stages.stage1_base_lm.token_budget -ne [int]$Stage1Tokens) { return $false }
    if ([int]$m.stages.stage2_curriculum.token_budget -ne [int]$Stage2Tokens) { return $false }
    if ([int]$m.stages.stage3_distill.token_budget -ne [int]$Stage3Tokens) { return $false }
    if ([int]$m.stages.stage4_instruction.token_budget -ne [int]$Stage4Tokens) { return $false }
    return $true
}

function Apply-TurboArgs([hashtable]$argTable, [string]$stageTag) {
    if (-not $TurboMode) {
        return
    }

    $seqBase = [int][Math]::Max(32, $SeqLen)
    $globalMin = [int][Math]::Max(32, [Math]::Min($TurboMinSeqLen, $seqBase))
    $maxFrozenGlobal = [int][Math]::Max(0, [Math]::Min($TurboMaxFrozenLayers, [Math]::Max(0, $NLayers - 1)))

    $minLen = $globalMin
    $seqWarmup = $TurboSeqWarmupRatio
    $depthWarmup = $TurboDepthWarmupRatio
    $maxFrozen = $maxFrozenGlobal

    switch ($stageTag) {
        "stage1" {
            $minLen = [int][Math]::Max($globalMin, [Math]::Floor($seqBase * 0.20))
            $seqWarmup = [double][Math]::Min(0.85, [Math]::Max($TurboSeqWarmupRatio, 0.65))
            $depthWarmup = [double][Math]::Min(0.60, [Math]::Max($TurboDepthWarmupRatio, 0.40))
            $maxFrozen = [int][Math]::Min([Math]::Max(0, $NLayers - 1), $maxFrozenGlobal + 2)
        }
        "stage2" {
            $minLen = [int][Math]::Max($globalMin, [Math]::Floor($seqBase * 0.30))
            $seqWarmup = [double][Math]::Min(0.75, [Math]::Max(0.45, $TurboSeqWarmupRatio))
            $depthWarmup = [double][Math]::Min(0.50, [Math]::Max(0.30, $TurboDepthWarmupRatio))
            $maxFrozen = $maxFrozenGlobal
        }
        "stage3" {
            $minLen = [int][Math]::Max($globalMin, [Math]::Floor($seqBase * 0.42))
            $seqWarmup = [double][Math]::Min(0.60, [Math]::Max(0.35, $TurboSeqWarmupRatio * 0.9))
            $depthWarmup = [double][Math]::Min(0.42, [Math]::Max(0.22, $TurboDepthWarmupRatio * 0.9))
            $maxFrozen = [int][Math]::Max(0, $maxFrozenGlobal - 1)
        }
        "stage4" {
            $minLen = [int][Math]::Max($globalMin, [Math]::Floor($seqBase * 0.60))
            $seqWarmup = [double][Math]::Min(0.45, [Math]::Max(0.20, $TurboSeqWarmupRatio * 0.7))
            $depthWarmup = [double][Math]::Min(0.30, [Math]::Max(0.12, $TurboDepthWarmupRatio * 0.6))
            $maxFrozen = [int][Math]::Max(0, $maxFrozenGlobal - 2)
        }
    }

    $argTable["TurboMode"] = $true
    $argTable["TurboMinSeqLen"] = [int][Math]::Max(32, [Math]::Min($minLen, $seqBase))
    $argTable["TurboSeqWarmupRatio"] = [double][Math]::Max(0.05, [Math]::Min($seqWarmup, 1.0))
    $argTable["TurboMaxFrozenLayers"] = [int][Math]::Max(0, $maxFrozen)
    $argTable["TurboDepthWarmupRatio"] = [double][Math]::Max(0.05, [Math]::Min($depthWarmup, 1.0))
    if ($TurboFreezeEmbeddings) {
        $argTable["TurboFreezeEmbeddings"] = $true
    }
}

$canReuse = $false
if ($SkipBuildIfReady -and (Test-StageFilesReady -stageDir $StagesDir) -and (Test-StageManifestMatches -stageDir $StagesDir)) {
    $canReuse = $true
}

if ($canReuse) {
    Write-Output "==> Reusing existing staged datasets (manifest matched)"
} else {
    Write-Output "==> Build 50M-token staged datasets"
    & $pythonExe scripts/build_50m_stage_datasets.py `
        --source_jsonl $SourceJsonl `
        --out_dir $StagesDir `
        --seed $Seed `
        --max_total_tokens $MaxTotalTokens `
        --stage1_tokens $Stage1Tokens `
        --stage2_tokens $Stage2Tokens `
        --stage3_tokens $Stage3Tokens `
        --stage4_tokens $Stage4Tokens `
        --license_policy $LicensePolicy
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$s1Data = Join-Path $StagesDir "stage1_base_lm.jsonl"
$s2Data = Join-Path $StagesDir "stage2_curriculum.jsonl"
$s3Data = Join-Path $StagesDir "stage3_distill.jsonl"
$s4Data = Join-Path $StagesDir "stage4_instruction.jsonl"

$s1Dir = Join-Path $RunDir "stage1_base_lm"
$s2Dir = Join-Path $RunDir "stage2_curriculum"
$s3Dir = Join-Path $RunDir "stage3_distill"
$s4Dir = Join-Path $RunDir "stage4_instruction"

$s1Steps = Get-StepsFromTokens -tokens $Stage1Tokens -seqLen $SeqLen -batchSize $BatchSize -gradAccum $GradAccumSteps
$s2Steps = Get-StepsFromTokens -tokens $Stage2Tokens -seqLen $SeqLen -batchSize $BatchSize -gradAccum $GradAccumSteps
$s3Steps = Get-StepsFromTokens -tokens $Stage3Tokens -seqLen $SeqLen -batchSize $BatchSize -gradAccum $GradAccumSteps
$s4Steps = Get-StepsFromTokens -tokens $Stage4Tokens -seqLen $SeqLen -batchSize $BatchSize -gradAccum $GradAccumSteps

$s1TargetSteps = $s1Steps
$s2TargetSteps = $s1TargetSteps + $s2Steps
$s3TargetSteps = $s2TargetSteps + $s3Steps
$s4TargetSteps = $s3TargetSteps + $s4Steps

$s1Warmup = [int][Math]::Max(50, [Math]::Ceiling($s1Steps * 0.03))
$s2Warmup = [int][Math]::Max(40, [Math]::Ceiling($s2Steps * 0.03))
$s3Warmup = [int][Math]::Max(40, [Math]::Ceiling($s3Steps * 0.03))
$s4Warmup = [int][Math]::Max(20, [Math]::Ceiling($s4Steps * 0.05))

$s2EmaDecay = 0.995
$s2ConsistencyLambda = 0.02
$s3EmaDecay = 0.995
$s3ConsistencyLambda = 0.02
if ($FastMode) {
    $s2EmaDecay = 0.0
    $s2ConsistencyLambda = 0.0
    $s3EmaDecay = 0.0
    $s3ConsistencyLambda = 0.0
}

Write-Output ("==> Stage delta steps: s1={0}, s2={1}, s3={2}, s4={3}" -f $s1Steps, $s2Steps, $s3Steps, $s4Steps)
Write-Output ("==> Stage target steps: s1={0}, s2={1}, s3={2}, s4={3}" -f $s1TargetSteps, $s2TargetSteps, $s3TargetSteps, $s4TargetSteps)
Write-Output ("==> Effective tokens/step: {0}" -f ($SeqLen * $BatchSize * $GradAccumSteps))
if ($TurboMode) {
    Write-Output ("==> Turbo4060 enabled: min_seq={0}, seq_warmup={1}, max_frozen_layers={2}, depth_warmup={3}, freeze_embeddings={4}" -f $TurboMinSeqLen, $TurboSeqWarmupRatio, $TurboMaxFrozenLayers, $TurboDepthWarmupRatio, $TurboFreezeEmbeddings)
}

Write-Output "==> Train Stage 1 (Base LM)"
$s1Args = @{
    DataPath = $s1Data
    OutputDir = $s1Dir
    TokenizerType = $TokenizerType
    Steps = $s1TargetSteps
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    SeqLen = $SeqLen
    DModel = $DModel
    NHeads = $NHeads
    NLayers = $NLayers
    MlpMult = $MlpMult
    Dropout = $Dropout
    NumWorkers = $NumWorkers
    LR = 0.0002
    WeightDecay = $WeightDecay
    WarmupSteps = $s1Warmup
    GradClip = $GradClip
    ValRatio = $ValRatio
    KoFocus = 0.8
    DocFocus = 0.3
    EMADecay = 0.0
    ConsistencyLambda = 0.0
    ConsistencyTemp = 1.5
    ConsistencyWarmup = 1000000
    EvalInterval = $EvalInterval
    SaveInterval = $SaveInterval
    EvalBatches = $EvalBatches
    SamplePrompt = "### Instruction`nExplain what a stack data structure is in Korean.`n`n### Response`n"
}
if ($TokenizerType -eq "spm") { $s1Args["TokenizerModel"] = $TokenizerModel }
Apply-TurboArgs -argTable $s1Args -stageTag "stage1"
if ($ActivationCheckpointing) { $s1Args["ActivationCheckpointing"] = $true }
if ($SaveStepCheckpoints) { $s1Args["SaveStepCheckpoints"] = $true }
if ($SkipSample) { $s1Args["SkipSample"] = $true }
if (-not $SaveBestCheckpoint) { $s1Args["NoBestCheckpoint"] = $true }
$s1Resume = Resolve-ResumeCheckpoint -stageDir $s1Dir
if ($s1Resume) {
    $s1Step = Get-CheckpointStep -ckptPath $s1Resume
    if ($s1Step -ge $s1TargetSteps) {
        Write-Output ("==> Stage 1 already complete (step={0} >= target={1}), skipping." -f $s1Step, $s1TargetSteps)
    } else {
        Write-Output ("==> Stage 1 resume from {0} (step={1})" -f $s1Resume, $s1Step)
        $s1Args["ResumeFrom"] = $s1Resume
        & "$PSScriptRoot/train_slm.ps1" @s1Args
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} else {
    & "$PSScriptRoot/train_slm.ps1" @s1Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$resumeS1 = Resolve-StageCheckpoint -stageDir $s1Dir

Write-Output "==> Train Stage 2 (Curriculum)"
$s2Args = @{
    DataPath = $s2Data
    OutputDir = $s2Dir
    TokenizerType = $TokenizerType
    Steps = $s2TargetSteps
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    SeqLen = $SeqLen
    DModel = $DModel
    NHeads = $NHeads
    NLayers = $NLayers
    MlpMult = $MlpMult
    Dropout = $Dropout
    NumWorkers = $NumWorkers
    LR = 0.00012
    WeightDecay = $WeightDecay
    WarmupSteps = $s2Warmup
    GradClip = $GradClip
    ValRatio = $ValRatio
    KoFocus = 1.1
    DocFocus = 0.6
    EMADecay = $s2EmaDecay
    ConsistencyLambda = $s2ConsistencyLambda
    ConsistencyTemp = 1.5
    ConsistencyWarmup = [int][Math]::Max(30, [Math]::Ceiling($s2Steps * 0.08))
    EvalInterval = $EvalInterval
    SaveInterval = $SaveInterval
    EvalBatches = $EvalBatches
    ResumeFrom = $resumeS1
    ResumeWeightsOnly = $true
    SamplePrompt = "### Instruction`nGive a concise TypeScript example of useState.`n`n### Response`n"
}
if ($TokenizerType -eq "spm") { $s2Args["TokenizerModel"] = $TokenizerModel }
Apply-TurboArgs -argTable $s2Args -stageTag "stage2"
if ($ActivationCheckpointing) { $s2Args["ActivationCheckpointing"] = $true }
if ($SaveStepCheckpoints) { $s2Args["SaveStepCheckpoints"] = $true }
if ($SkipSample) { $s2Args["SkipSample"] = $true }
if (-not $SaveBestCheckpoint) { $s2Args["NoBestCheckpoint"] = $true }
$s2Resume = Resolve-ResumeCheckpoint -stageDir $s2Dir
if ($s2Resume) {
    $s2Step = Get-CheckpointStep -ckptPath $s2Resume
    if ($s2Step -ge $s2TargetSteps) {
        Write-Output ("==> Stage 2 already complete (step={0} >= target={1}), skipping." -f $s2Step, $s2TargetSteps)
    } else {
        Write-Output ("==> Stage 2 resume from {0} (step={1})" -f $s2Resume, $s2Step)
        $s2Args["ResumeFrom"] = $s2Resume
        $s2Args["ResumeWeightsOnly"] = $false
        & "$PSScriptRoot/train_slm.ps1" @s2Args
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} else {
    & "$PSScriptRoot/train_slm.ps1" @s2Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$resumeS2 = Resolve-StageCheckpoint -stageDir $s2Dir

Write-Output "==> Train Stage 3 (Distillation)"
$s3Args = @{
    DataPath = $s3Data
    OutputDir = $s3Dir
    TokenizerType = $TokenizerType
    Steps = $s3TargetSteps
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    SeqLen = $SeqLen
    DModel = $DModel
    NHeads = $NHeads
    NLayers = $NLayers
    MlpMult = $MlpMult
    Dropout = $Dropout
    NumWorkers = $NumWorkers
    LR = 0.00008
    WeightDecay = $WeightDecay
    WarmupSteps = $s3Warmup
    GradClip = $GradClip
    ValRatio = $ValRatio
    KoFocus = 1.2
    DocFocus = 0.8
    EMADecay = $s3EmaDecay
    ConsistencyLambda = $s3ConsistencyLambda
    ConsistencyTemp = 1.5
    ConsistencyWarmup = [int][Math]::Max(30, [Math]::Ceiling($s3Steps * 0.05))
    EvalInterval = $EvalInterval
    SaveInterval = $SaveInterval
    EvalBatches = $EvalBatches
    ResumeFrom = $resumeS2
    ResumeWeightsOnly = $true
    SamplePrompt = "### Instruction`nList key identifiers from this code snippet.`n`n### Response`n"
}
if ($TokenizerType -eq "spm") { $s3Args["TokenizerModel"] = $TokenizerModel }
Apply-TurboArgs -argTable $s3Args -stageTag "stage3"
if ($ActivationCheckpointing) { $s3Args["ActivationCheckpointing"] = $true }
if ($SaveStepCheckpoints) { $s3Args["SaveStepCheckpoints"] = $true }
if ($SkipSample) { $s3Args["SkipSample"] = $true }
if (-not $SaveBestCheckpoint) { $s3Args["NoBestCheckpoint"] = $true }
$s3Resume = Resolve-ResumeCheckpoint -stageDir $s3Dir
if ($s3Resume) {
    $s3Step = Get-CheckpointStep -ckptPath $s3Resume
    if ($s3Step -ge $s3TargetSteps) {
        Write-Output ("==> Stage 3 already complete (step={0} >= target={1}), skipping." -f $s3Step, $s3TargetSteps)
    } else {
        Write-Output ("==> Stage 3 resume from {0} (step={1})" -f $s3Resume, $s3Step)
        $s3Args["ResumeFrom"] = $s3Resume
        $s3Args["ResumeWeightsOnly"] = $false
        & "$PSScriptRoot/train_slm.ps1" @s3Args
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} else {
    & "$PSScriptRoot/train_slm.ps1" @s3Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$resumeS3 = Resolve-StageCheckpoint -stageDir $s3Dir

Write-Output "==> Train Stage 4 (Instruction)"
$s4Args = @{
    DataPath = $s4Data
    OutputDir = $s4Dir
    TokenizerType = $TokenizerType
    Steps = $s4TargetSteps
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    SeqLen = $SeqLen
    DModel = $DModel
    NHeads = $NHeads
    NLayers = $NLayers
    MlpMult = $MlpMult
    Dropout = 0.08
    NumWorkers = $NumWorkers
    LR = 0.00003
    WeightDecay = $WeightDecay
    WarmupSteps = $s4Warmup
    GradClip = $GradClip
    ValRatio = $ValRatio
    KoFocus = 1.0
    DocFocus = 0.5
    EMADecay = 0.0
    ConsistencyLambda = 0.0
    ConsistencyTemp = 1.5
    ConsistencyWarmup = 1000000
    EvalInterval = [int][Math]::Max(50, [Math]::Floor($EvalInterval / 2))
    SaveInterval = [int][Math]::Max(100, [Math]::Floor($SaveInterval / 2))
    EvalBatches = $EvalBatches
    ResumeFrom = $resumeS3
    ResumeWeightsOnly = $true
    SamplePrompt = "### Instruction`nReact useState example in one short paragraph.`n`n### Response`n"
}
if ($TokenizerType -eq "spm") { $s4Args["TokenizerModel"] = $TokenizerModel }
Apply-TurboArgs -argTable $s4Args -stageTag "stage4"
if ($ActivationCheckpointing) { $s4Args["ActivationCheckpointing"] = $true }
if ($SaveStepCheckpoints) { $s4Args["SaveStepCheckpoints"] = $true }
if ($SkipSample) { $s4Args["SkipSample"] = $true }
if (-not $SaveBestCheckpoint) { $s4Args["NoBestCheckpoint"] = $true }
$s4Resume = Resolve-ResumeCheckpoint -stageDir $s4Dir
if ($s4Resume) {
    $s4Step = Get-CheckpointStep -ckptPath $s4Resume
    if ($s4Step -ge $s4TargetSteps) {
        Write-Output ("==> Stage 4 already complete (step={0} >= target={1}), skipping." -f $s4Step, $s4TargetSteps)
    } else {
        Write-Output ("==> Stage 4 resume from {0} (step={1})" -f $s4Resume, $s4Step)
        $s4Args["ResumeFrom"] = $s4Resume
        $s4Args["ResumeWeightsOnly"] = $false
        & "$PSScriptRoot/train_slm.ps1" @s4Args
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} else {
    & "$PSScriptRoot/train_slm.ps1" @s4Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$finalCkpt = Resolve-StageCheckpoint -stageDir $s4Dir
Write-Output ("==> 4-stage training finished. final_checkpoint={0}" -f $finalCkpt)
