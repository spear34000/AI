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
    [switch]$UseUGC,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cpu",
    [string]$SystemPrompt = ""
)

$chatArgs = @{
    Checkpoint = $Checkpoint
    MaxNewTokens = $MaxNewTokens
    Temperature = $Temperature
    TopK = $TopK
    TopP = $TopP
    RepetitionPenalty = $RepetitionPenalty
    HistoryTurns = $HistoryTurns
    Device = $Device
    SystemPrompt = $SystemPrompt
}

if ($Prompt -and $Prompt.Trim().Length -gt 0) {
    $chatArgs["Prompt"] = $Prompt
}
if ($UseEMA) {
    $chatArgs["UseEMA"] = $true
}
if ($NoEMA) {
    $chatArgs["NoEMA"] = $true
}

& "$PSScriptRoot/chat_slm.ps1" @chatArgs
$exitCode = $LASTEXITCODE

if ($UseUGC) {
    Write-Output "UseUGC mode is deprecated and ignored in current mobile stack."
}

Write-Output "Mobile chat finished. cpu_threads=2, perf_cap=20%"

if ($exitCode -ne 0) {
    exit $exitCode
}
