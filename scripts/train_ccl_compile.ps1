param(
    [string]$BaseCheckpoint = "artifacts_ondevice_best/slm_ondevice_fp16.pt",
    [string]$SpecPath = "data/ccl_specbook_v1.jsonl",
    [string]$DataPath = "data/slm_mit_unified_v4.jsonl",
    [string]$OutputDir = "artifacts_ccl2_compile",
    [int]$Steps = 80,
    [int]$MaxSpecs = 1800,
    [int]$MineSeeds = 32,
    [int]$MutationsPerSeed = 4,
    [int]$CounterexamplesPerStep = 8,
    [int]$CompileBatchSize = 6,
    [int]$VerifyInterval = 4,
    [int]$VerifySpecs = 300,
    [double]$TargetHardPass = 0.95,
    [int]$PatchTopLayers = 4,
    [string]$PatchTargets = "qkv,proj,mlp_in,mlp_out",
    [int]$LoraRank = 8,
    [double]$LoraAlpha = 16.0,
    [double]$LoraDropout = 0.0,
    [int]$CgIters = 10,
    [double]$CgTol = 1e-4,
    [double]$Damping = 0.08,
    [double]$StepScale = 0.8,
    [double]$MaxUpdateNorm = 0.08,
    [int]$LedgerMaxBasis = 24,
    [int]$LedgerRefreshInterval = 3,
    [int]$LedgerRefreshSamples = 6,
    [double]$NoveltyThreshold = 0.92,
    [int]$SaveInterval = 8,
    [int]$Seed = 42,
    [int]$SeqLen = 0,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [string]$Mutators = "identity,boundary,constraint_clash,reorder,ambiguity,whitespace,length_cap",
    [int]$MaxNewTokens = 96,
    [string]$ResumePatch = ""
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

function Test-PythonWithTorch([string]$exePath) {
    if (-not (Test-PythonExe -exePath $exePath)) {
        return $false
    }
    try {
        & $exePath -c "import torch; print(torch.__version__)" *> $null
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
        if (Test-PythonWithTorch -exePath $c) {
            return $c
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-PythonWithTorch -exePath $cmd.Source)) {
        return $cmd.Source
    }

    throw "No usable python executable with torch found."
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Resolve-PythonExe -repoRoot $repoRoot

$argsList = @(
    "scripts/train_ccl_compile.py",
    "--base_checkpoint", $BaseCheckpoint,
    "--spec_path", $SpecPath,
    "--data_path", $DataPath,
    "--output_dir", $OutputDir,
    "--steps", $Steps,
    "--max_specs", $MaxSpecs,
    "--mine_seeds", $MineSeeds,
    "--mutations_per_seed", $MutationsPerSeed,
    "--counterexamples_per_step", $CounterexamplesPerStep,
    "--compile_batch_size", $CompileBatchSize,
    "--verify_interval", $VerifyInterval,
    "--verify_specs", $VerifySpecs,
    "--target_hard_pass", $TargetHardPass,
    "--patch_top_layers", $PatchTopLayers,
    "--patch_targets", $PatchTargets,
    "--lora_rank", $LoraRank,
    "--lora_alpha", $LoraAlpha,
    "--lora_dropout", $LoraDropout,
    "--cg_iters", $CgIters,
    "--cg_tol", $CgTol,
    "--damping", $Damping,
    "--step_scale", $StepScale,
    "--max_update_norm", $MaxUpdateNorm,
    "--ledger_max_basis", $LedgerMaxBasis,
    "--ledger_refresh_interval", $LedgerRefreshInterval,
    "--ledger_refresh_samples", $LedgerRefreshSamples,
    "--novelty_threshold", $NoveltyThreshold,
    "--save_interval", $SaveInterval,
    "--seed", $Seed,
    "--seq_len", $SeqLen,
    "--device", $Device,
    "--mutators", $Mutators,
    "--max_new_tokens", $MaxNewTokens
)

if ($ResumePatch -and $ResumePatch.Trim().Length -gt 0) {
    $argsList += @("--resume_patch", $ResumePatch)
}

Push-Location $repoRoot
try {
    & $pythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
