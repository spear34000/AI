param(
    [string]$Patch = "artifacts_ccl2_compile/ccl_patch_best.pt",
    [string]$BaseCheckpoint = "",
    [double]$PatchScale = 0.35,
    [string]$Prompt = "",
    [string]$SystemPrompt = "",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$HistoryTurns = 6,
    [int]$MaxNewTokens = 180,
    [double]$Temperature = 0.0,
    [int]$TopK = 0,
    [double]$TopP = 1.0,
    [double]$RepetitionPenalty = 1.12,
    [switch]$ShowMeta
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
    "scripts/chat_ccl_patch.py",
    "--patch", $Patch,
    "--patch_scale", $PatchScale,
    "--device", $Device,
    "--history_turns", $HistoryTurns,
    "--max_new_tokens", $MaxNewTokens,
    "--temperature", $Temperature,
    "--top_k", $TopK,
    "--top_p", $TopP,
    "--repetition_penalty", $RepetitionPenalty
)

if ($BaseCheckpoint -and $BaseCheckpoint.Trim().Length -gt 0) {
    $argsList += @("--base_checkpoint", $BaseCheckpoint)
}
if ($Prompt -and $Prompt.Trim().Length -gt 0) {
    $argsList += @("--prompt", $Prompt)
}
if ($SystemPrompt -and $SystemPrompt.Trim().Length -gt 0) {
    $argsList += @("--system_prompt", $SystemPrompt)
}
if ($ShowMeta) {
    $argsList += @("--show_meta")
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
