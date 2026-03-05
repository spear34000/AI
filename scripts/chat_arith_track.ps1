param(
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [string]$Device = "cuda",
    [int]$MaxNewTokens = 128,
    [double]$Temperature = 0.0,
    [int]$TopK = 1,
    [double]$TopP = 1.0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = 'C:\Program Files\Python311\python.exe'
$ckpt = 'artifacts_teacher_arith_pure_mix_fromscratch_v3_runA\slm_last.pt'

& $py scripts\chat_slm.py `
  --checkpoint $ckpt `
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
