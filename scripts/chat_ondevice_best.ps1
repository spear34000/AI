param(
    [string]$ManifestPath = "artifacts_ondevice_best/manifest.json",
    [string]$Prompt = "",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cpu",
    [int]$MaxNewTokens = 180,
    [double]$Temperature = 0.0,
    [int]$TopK = 0,
    [double]$TopP = 1.0,
    [double]$RepetitionPenalty = 1.12,
    [int]$HistoryTurns = 6,
    [string]$SystemPrompt = "",
    [string]$SessionId = "ondevice_best",
    [switch]$DisableRetrieval,
    [switch]$NoInt8
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

Push-Location $repoRoot
try {
    if (-not (Test-Path $ManifestPath)) {
        throw "manifest not found: $ManifestPath. Run scripts/build_best_ondevice.ps1 first."
    }

    $manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $checkpoint = [string]$manifest.output_checkpoint
    if (-not $checkpoint -or -not (Test-Path $checkpoint)) {
        throw "output checkpoint not found: $checkpoint"
    }

    $argsList = @(
        "scripts/chat_slm.py",
        "--checkpoint", $checkpoint,
        "--device", $Device,
        "--max_new_tokens", $MaxNewTokens,
        "--temperature", $Temperature,
        "--top_k", $TopK,
        "--top_p", $TopP,
        "--repetition_penalty", $RepetitionPenalty,
        "--history_turns", $HistoryTurns,
        "--session_id", $SessionId,
        "--agent_mode", "auto"
    )

    if ($Prompt -and $Prompt.Trim().Length -gt 0) {
        $argsList += @("--prompt", $Prompt)
    }
    if ($SystemPrompt -and $SystemPrompt.Trim().Length -gt 0) {
        $argsList += @("--system_prompt", $SystemPrompt)
    }
    if ($DisableRetrieval) {
        $argsList += @("--disable_retrieval")
    }
    if ($Device -eq "cpu" -and (-not $NoInt8)) {
        $argsList += @("--quantize_int8")
    }

    & $pythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
