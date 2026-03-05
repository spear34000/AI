param(
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = 'C:\Program Files\Python311\python.exe'
$general = 'artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2\slm_last.pt'
$arith = 'artifacts_teacher_arith_pure_mix_fromscratch_v3_runA\slm_last.pt'

& $py .\scripts\eval_prompt_suite_v1.py `
  --checkpoints $general $arith `
  --suite data\eval_prompt_suite_v2.json `
  --device $Device `
  --max_new_tokens 96 `
  --out_json dual_track_eval_prompt_suite_v2.json

& $py .\scripts\eval_logic_exact_v1.py `
  --checkpoints $arith `
  --data_path data\teacher_arith_pure_mix_v3_eval.jsonl `
  --limit 512 `
  --device $Device `
  --mode inprocess `
  --out_json dual_track_eval_arith_v3.json
