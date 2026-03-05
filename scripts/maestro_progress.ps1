param(
  [string]$PythonPath = "C:\Program Files\Python311\python.exe"
)

$ErrorActionPreference = "Stop"

function Get-StepFromCheckpoint {
  param([string]$CheckpointPath, [string]$PythonExe)
  if (-not (Test-Path $CheckpointPath)) { return $null }
  $code = @"
import json
import torch
path = r'''$CheckpointPath'''
ck = torch.load(path, map_location='cpu', weights_only=True)
print(int(ck.get('step', 0)))
"@
  $step = & $PythonExe -c $code 2>$null
  if (-not $step) { return $null }
  return [int]$step
}

$trainProcs = Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*scripts/train_slm.py*" }

if (-not $trainProcs) {
  Write-Output "No active train_slm.py process found."
  exit 0
}

foreach ($p in $trainProcs) {
  $cmd = [string]$p.CommandLine
  $procId = [int]$p.ProcessId

  $outDir = ""
  $steps = 0
  if ($cmd -match "--output_dir\s+([^\s]+)") { $outDir = $Matches[1] }
  if ($cmd -match "--steps\s+([0-9]+)") { $steps = [int]$Matches[1] }
  if (-not $outDir) { continue }

  $bestPath = Join-Path $outDir "slm_best.pt"
  $lastPath = Join-Path $outDir "slm_last.pt"
  $stepBest = Get-StepFromCheckpoint -CheckpointPath $bestPath -PythonExe $PythonPath
  $stepLast = Get-StepFromCheckpoint -CheckpointPath $lastPath -PythonExe $PythonPath

  $step = $null
  if ($stepLast -ne $null -and $stepBest -ne $null) {
    $step = [Math]::Max($stepLast, $stepBest)
  } elseif ($stepLast -ne $null) {
    $step = $stepLast
  } elseif ($stepBest -ne $null) {
    $step = $stepBest
  }

  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  $elapsedSec = $null
  $etaMin = $null
  if ($proc) {
    $elapsedSec = [Math]::Max(1.0, (Get-Date).Subtract($proc.StartTime).TotalSeconds)
    if ($step -ne $null -and $step -gt 0 -and $steps -gt $step) {
      $secPerStep = $elapsedSec / [double]$step
      $etaMin = [Math]::Round((($steps - $step) * $secPerStep) / 60.0, 1)
    }
  }

  [pscustomobject]@{
    pid = $procId
    output_dir = $outDir
    step = $step
    target_steps = $steps
    progress_pct = $(if ($step -ne $null -and $steps -gt 0) { [Math]::Round((100.0 * $step / $steps), 2) } else { $null })
    elapsed_min = $(if ($elapsedSec -ne $null) { [Math]::Round($elapsedSec / 60.0, 1) } else { $null })
    eta_min = $etaMin
  } | ConvertTo-Json -Compress
}
