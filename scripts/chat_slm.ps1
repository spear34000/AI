param(
    [string]$Checkpoint = "artifacts_slm_mit_unified_v4/slm_best.pt",
    [string]$Prompt = "",
    [int]$MaxNewTokens = 180,
    [double]$Temperature = 0.0,
    [int]$TopK = 0,
    [double]$TopP = 1.0,
    [double]$RepetitionPenalty = 1.12,
    [int]$HistoryTurns = 6,
    [switch]$UseEMA,
    [switch]$NoEMA,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [string]$SystemPrompt = "",
    [ValidateSet("none", "min_qa_ko")]
    [string]$SystemPreset = "none",
    [ValidateSet("off", "legacy")]
    [string]$HeuristicMode = "off",
    [ValidateSet("off", "balanced", "strict")]
    [string]$ZeroShotMode = "off",
    [ValidateSet("off", "auto", "triad")]
    [string]$AgentMode = "off",
    [string]$ToolCachePath = "data/tool_knowledge_cache_v3_clean.jsonl",
    [double]$ToolLookupTimeout = 4.0,
    [switch]$DisableWebToolLookup,
    [switch]$DisableQualityRerank,
    [switch]$ForceRaw
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath, [bool]$RequireTorch = $false) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($RequireTorch) {
            & $exePath -c "import torch" *> $null
            if ($LASTEXITCODE -ne 0) { return $false }
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

$argsList = @(
    "scripts/chat_slm.py",
    "--checkpoint", $Checkpoint,
    "--max_new_tokens", $MaxNewTokens,
    "--temperature", $Temperature,
    "--top_k", $TopK,
    "--top_p", $TopP,
    "--repetition_penalty", $RepetitionPenalty,
    "--history_turns", $HistoryTurns,
    "--device", $Device
)

if ($Prompt -and $Prompt.Trim().Length -gt 0) {
    $argsList += @("--prompt", $Prompt)
}

$applyEMA = $true
if ($NoEMA) {
    $applyEMA = $false
}
if ($UseEMA) {
    $applyEMA = $true
}
if ($applyEMA) {
    $argsList += @("--use_ema")
}

if ($SystemPrompt -and $SystemPrompt.Trim().Length -gt 0) {
    $argsList += @("--system_prompt", $SystemPrompt)
}
if ($SystemPreset -and $SystemPreset -ne "none") {
    $argsList += @("--system_preset", $SystemPreset)
}
if ($HeuristicMode) {
    $argsList += @("--heuristic_mode", $HeuristicMode)
}
if ($ZeroShotMode) {
    $argsList += @("--zero_shot_mode", $ZeroShotMode)
}
if ($AgentMode) {
    $argsList += @("--agent_mode", $AgentMode)
}
if ($ToolCachePath -and $ToolCachePath.Trim().Length -gt 0) {
    $argsList += @("--tool_cache_path", $ToolCachePath)
}
if ($ToolLookupTimeout -gt 0) {
    $argsList += @("--tool_lookup_timeout", $ToolLookupTimeout)
}
if ($DisableWebToolLookup) {
    $argsList += @("--disable_web_tool_lookup")
}
if ($DisableQualityRerank) {
    $argsList += @("--disable_quality_rerank")
}
if ($ForceRaw) {
    $argsList += @("--force_raw")
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
