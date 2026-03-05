param(
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [string]$Device = "cuda",
    [int]$MaxNewTokens = 128,
    [string]$GeneralCheckpoint = "artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt",
    [string]$LogicCheckpoint = "artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage3/slm_last.pt"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = 'C:\Program Files\Python311\python.exe'
$logicMarkers = @(
    '가장 작은', '확률', '수열', '모든', '어떤', 'A는 B보다', 'B는 C보다'
)
$useLogic = $false
foreach ($marker in $logicMarkers) {
    if ($Prompt.Contains($marker)) {
        $useLogic = $true
        break
    }
}

$ckpt = if ($useLogic) { $LogicCheckpoint } else { $GeneralCheckpoint }
$route = if ($useLogic) { 'logic' } else { 'general' }
Write-Host "[dev-broker] route=$route checkpoint=$ckpt"

& $py scripts\chat_slm.py `
  --checkpoint $ckpt `
  --router single `
  --prompt $Prompt `
  --device $Device `
  --temperature 0 `
  --top_k 1 `
  --top_p 1.0 `
  --max_new_tokens $MaxNewTokens `
  --disable_quality_rerank `
  --zero_shot_mode off `
  --heuristic_mode off `
  --agent_mode off `
  --force_raw
