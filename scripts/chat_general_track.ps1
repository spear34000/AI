param(
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [string]$Checkpoint = "artifacts_mainline_suite_fix_v1_runA/slm_best.pt",
    [string]$Device = "cuda",
    [int]$MaxNewTokens = 128,
    [double]$Temperature = 0.0,
    [int]$TopK = 1,
    [double]$TopP = 1.0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$pyCandidates = @(
    "C:\Program Files\Python311\python.exe",
    ".\.venv\Scripts\python.exe"
)
$py = $null
foreach ($cand in $pyCandidates) {
    if (-not (Test-Path $cand)) { continue }
    if ($Device -eq "cuda") {
        try {
            $ok = & $cand -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
            if ($ok -and $ok.Trim() -eq "1") {
                $py = $cand
                break
            }
        } catch {
            continue
        }
    } else {
        $py = $cand
        break
    }
}
if (-not $py) {
    throw "No suitable Python interpreter found for device=$Device"
}

$ck = $Checkpoint
if (-not (Test-Path $ck)) {
    throw "checkpoint not found: $ck"
}

$env:PYTHONIOENCODING = "utf-8"
& $py scripts/chat_slm.py `
    --checkpoint $ck `
    --router single `
    --prompt $Prompt `
    --device $Device `
    --temperature $Temperature `
    --top_k $TopK `
    --top_p $TopP `
    --max_new_tokens $MaxNewTokens `
    --disable_quality_rerank `
    --zero_shot_mode off `
    --heuristic_mode off `
    --agent_mode off `
    --force_raw
