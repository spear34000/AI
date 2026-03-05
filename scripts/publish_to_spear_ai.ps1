param(
  [Parameter(Mandatory = $true)]
  [string]$Message,
  [string]$CheckpointPath = "",
  [switch]$AutoPickCheckpoint,
  [switch]$DryRun,
  [switch]$Force,
  [string]$ManifestPath = "configs/publish_manifest_mainline_v1.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $script:PSNativeCommandUseErrorActionPreference = $false
}

function Invoke-Git {
  param(
    [Parameter(Mandatory = $true)][string[]]$Args,
    [switch]$IgnoreError
  )
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $output = & git @Args 2>&1
  } finally {
    $ErrorActionPreference = $prevEap
  }
  $code = $LASTEXITCODE
  $text = [string]($output -join "`n")
  if ($code -ne 0 -and -not $IgnoreError) {
    throw "git $($Args -join ' ') failed (exit=$code)`n$text"
  }
  return [pscustomobject]@{
    ExitCode = $code
    Output = $text
  }
}

function Resolve-RepoRoot {
  return (Split-Path -Parent $PSScriptRoot)
}

function Resolve-FullPath {
  param(
    [Parameter(Mandatory = $true)][string]$PathValue,
    [Parameter(Mandatory = $true)][string]$RepoRoot
  )
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function To-RepoRelative {
  param(
    [Parameter(Mandatory = $true)][string]$FullPath,
    [Parameter(Mandatory = $true)][string]$RepoRoot
  )
  $root = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
  $full = [System.IO.Path]::GetFullPath($FullPath)
  if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "path is outside repo: $full"
  }
  if ($full.Length -eq $root.Length) { return "." }
  $rel = $full.Substring($root.Length).TrimStart('\')
  return $rel.Replace('\', '/')
}

function Load-Manifest {
  param([Parameter(Mandatory = $true)][string]$PathValue)
  if (-not (Test-Path -LiteralPath $PathValue)) {
    throw "manifest not found: $PathValue"
  }
  return Get-Content -LiteralPath $PathValue -Raw -Encoding utf8 | ConvertFrom-Json
}

function Ensure-Bootstrap {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)]$Manifest,
    [switch]$ForceUpdate
  )
  Push-Location $RepoRoot
  try {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot ".git"))) {
      $null = Invoke-Git -Args @("init")
    }

    $remoteName = "origin"
    $desiredRemote = [string]$Manifest.remote_url
    $remoteQuery = Invoke-Git -Args @("remote", "get-url", $remoteName) -IgnoreError
    $currentRemote = ""
    if ($remoteQuery.ExitCode -eq 0) {
      $currentRemote = $remoteQuery.Output.Trim()
    }
    if ([string]::IsNullOrWhiteSpace($currentRemote)) {
      $null = Invoke-Git -Args @("remote", "add", $remoteName, $desiredRemote)
    } elseif ($currentRemote -ne $desiredRemote) {
      if (-not $ForceUpdate) {
        throw "origin remote differs. current=$currentRemote desired=$desiredRemote. Use -Force to update."
      }
      $null = Invoke-Git -Args @("remote", "set-url", $remoteName, $desiredRemote)
    }

    $null = Invoke-Git -Args @("lfs", "install", "--local")
    $null = Invoke-Git -Args @("fetch", $remoteName, [string]$Manifest.branch)

    $branch = [string]$Manifest.branch
    $hasRemoteBranch = (Invoke-Git -Args @("show-ref", "--verify", "--quiet", "refs/remotes/$remoteName/$branch") -IgnoreError).ExitCode -eq 0
    $hasLocalBranch = (Invoke-Git -Args @("show-ref", "--verify", "--quiet", "refs/heads/$branch") -IgnoreError).ExitCode -eq 0

    if (-not $hasLocalBranch) {
      if ($hasRemoteBranch) {
        $null = Invoke-Git -Args @("checkout", "-B", $branch, "$remoteName/$branch")
      } else {
        $null = Invoke-Git -Args @("checkout", "-B", $branch)
      }
    } else {
      $null = Invoke-Git -Args @("checkout", $branch)
    }
  } finally {
    Pop-Location
  }
}

function Get-WinnerFromPromotion {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)]$Manifest
  )
  $path = Resolve-FullPath -PathValue ([string]$Manifest.promotion_report_path) -RepoRoot $RepoRoot
  if (-not (Test-Path -LiteralPath $path)) { return $null }
  $payload = Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
  if ($null -eq $payload) { return $null }
  $winner = [string]$payload.winner
  if ([string]::IsNullOrWhiteSpace($winner)) { return $null }
  $full = Resolve-FullPath -PathValue $winner -RepoRoot $RepoRoot
  if (Test-Path -LiteralPath $full) { return $full }
  return $null
}

function Get-BestFromEval {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)]$Manifest
  )
  $path = Resolve-FullPath -PathValue ([string]$Manifest.eval_report_path) -RepoRoot $RepoRoot
  if (-not (Test-Path -LiteralPath $path)) { return $null }
  $payload = Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
  if ($null -eq $payload -or $null -eq $payload.exact_runs) { return $null }

  $scores = @{}
  foreach ($run in $payload.exact_runs) {
    $outJson = [string]$run.out_json
    if ([string]::IsNullOrWhiteSpace($outJson)) { continue }
    $outPath = Resolve-FullPath -PathValue $outJson -RepoRoot $RepoRoot
    if (-not (Test-Path -LiteralPath $outPath)) { continue }
    $res = Get-Content -LiteralPath $outPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($null -eq $res -or $null -eq $res.results) { continue }
    foreach ($r in $res.results) {
      $ck = [string]$r.checkpoint
      if ([string]::IsNullOrWhiteSpace($ck)) { continue }
      $full = Resolve-FullPath -PathValue $ck -RepoRoot $RepoRoot
      if (-not (Test-Path -LiteralPath $full)) { continue }
      $exact = 0.0
      try { $exact = [double]$r.exact_match } catch { $exact = 0.0 }
      if (-not $scores.ContainsKey($full) -or $exact -gt $scores[$full]) {
        $scores[$full] = $exact
      }
    }
  }
  if ($scores.Count -eq 0) { return $null }
  return ($scores.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1).Key
}

function Select-Checkpoint {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)]$Manifest,
    [string]$ExplicitPath,
    [switch]$UseAutoPick
  )
  if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
    $full = Resolve-FullPath -PathValue $ExplicitPath -RepoRoot $RepoRoot
    if (-not (Test-Path -LiteralPath $full)) {
      throw "checkpoint not found: $full"
    }
    return $full
  }
  if (-not $UseAutoPick) {
    return $null
  }
  $fromPromotion = Get-WinnerFromPromotion -RepoRoot $RepoRoot -Manifest $Manifest
  if ($fromPromotion) { return $fromPromotion }
  return (Get-BestFromEval -RepoRoot $RepoRoot -Manifest $Manifest)
}

function Is-Excluded {
  param(
    [Parameter(Mandatory = $true)][string]$RelativePath,
    [Parameter(Mandatory = $true)]$Patterns
  )
  foreach ($raw in $Patterns) {
    $pat = ([string]$raw).Replace('\', '/').Replace('**', '*')
    if ($RelativePath -like $pat) { return $true }
  }
  return $false
}

function Get-FilesToStage {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)]$Manifest,
    [string]$SelectedCheckpointRelative
  )
  $files = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)

  $addFile = {
    param([string]$full)
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { return }
    $rel = To-RepoRelative -FullPath $full -RepoRoot $RepoRoot
    if (Is-Excluded -RelativePath $rel -Patterns $Manifest.exclude_globs) { return }
    [void]$files.Add($rel)
  }

  foreach ($rootRel in $Manifest.include_roots) {
    $rootPath = Resolve-FullPath -PathValue ([string]$rootRel) -RepoRoot $RepoRoot
    if (-not (Test-Path -LiteralPath $rootPath)) { continue }
    Get-ChildItem -LiteralPath $rootPath -Recurse -File | ForEach-Object { & $addFile $_.FullName }
  }

  Push-Location $RepoRoot
  try {
    foreach ($glob in $Manifest.include_globs) {
      Get-ChildItem -Path ([string]$glob) -File -ErrorAction SilentlyContinue | ForEach-Object { & $addFile $_.FullName }
    }
  } finally {
    Pop-Location
  }

  if (-not [string]::IsNullOrWhiteSpace($SelectedCheckpointRelative)) {
    [void]$files.Add($SelectedCheckpointRelative)
  }

  return @($files | Sort-Object)
}

function Add-FileBatch {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string[]]$FileList,
    [switch]$UseForce
  )
  if ($FileList.Count -eq 0) { return }
  Push-Location $RepoRoot
  try {
    $filtered = New-Object System.Collections.Generic.List[string]
    foreach ($f in $FileList) {
      $exists = Test-Path -LiteralPath (Join-Path $RepoRoot $f) -PathType Leaf
      if (-not $exists) { continue }
      if (-not $UseForce) {
        $ignored = Invoke-Git -Args @("check-ignore", "--quiet", "--", $f) -IgnoreError
        if ($ignored.ExitCode -eq 0) { continue }
      }
      [void]$filtered.Add($f)
    }
    if ($filtered.Count -eq 0) { return }

    $batchSize = 200
    for ($i = 0; $i -lt $filtered.Count; $i += $batchSize) {
      $count = [Math]::Min($batchSize, $filtered.Count - $i)
      $chunk = $filtered[$i..($i + $count - 1)]
      if ($UseForce) {
        Invoke-Git -Args (@("add", "-f", "--") + $chunk)
      } else {
        Invoke-Git -Args (@("add", "--") + $chunk)
      }
    }
  } finally {
    Pop-Location
  }
}

$repoRoot = Resolve-RepoRoot
$manifestFullPath = Resolve-FullPath -PathValue $ManifestPath -RepoRoot $repoRoot
$manifest = Load-Manifest -PathValue $manifestFullPath

Ensure-Bootstrap -RepoRoot $repoRoot -Manifest $manifest -ForceUpdate:$Force

$selectedCheckpoint = Select-Checkpoint -RepoRoot $repoRoot -Manifest $manifest -ExplicitPath $CheckpointPath -UseAutoPick:$AutoPickCheckpoint
$selectedRel = $null
if (-not [string]::IsNullOrWhiteSpace($selectedCheckpoint)) {
  $selectedRel = To-RepoRelative -FullPath $selectedCheckpoint -RepoRoot $repoRoot
  Push-Location $repoRoot
  try {
    $null = Invoke-Git -Args @("lfs", "track", "--", $selectedRel)
  } finally {
    Pop-Location
  }
}

$files = Get-FilesToStage -RepoRoot $repoRoot -Manifest $manifest -SelectedCheckpointRelative $selectedRel
if ($files.Count -eq 0) {
  throw "no files selected for staging"
}

if ($DryRun) {
  $dry = [ordered]@{
    dry_run = $true
    repo_root = $repoRoot
    branch = [string]$manifest.branch
    selected_checkpoint = $selectedRel
    file_count = $files.Count
    files = $files
  }
  $dry | ConvertTo-Json -Depth 10
  exit 0
}

$forceFiles = @()
if (-not [string]::IsNullOrWhiteSpace($selectedRel)) {
  $forceFiles += $selectedRel
}
$normalFiles = @($files | Where-Object { $forceFiles -notcontains $_ })

Add-FileBatch -RepoRoot $repoRoot -FileList $normalFiles
Add-FileBatch -RepoRoot $repoRoot -FileList $forceFiles -UseForce

Push-Location $repoRoot
try {
  $staged = Invoke-Git -Args @("diff", "--cached", "--name-only")
  $stagedFiles = @($staged.Output -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($stagedFiles.Count -eq 0) {
    Write-Output "No staged changes. Nothing to commit."
    exit 0
  }

  Invoke-Git -Args @("commit", "-m", $Message)

  $pull = Invoke-Git -Args @("pull", "--rebase", "origin", [string]$manifest.branch) -IgnoreError
  if ($pull.ExitCode -ne 0 -and -not $Force) {
    throw "git pull --rebase failed. Resolve conflicts or rerun with -Force.`n$($pull.Output)"
  }

  Invoke-Git -Args @("push", "origin", [string]$manifest.branch)
  $sha = (Invoke-Git -Args @("rev-parse", "HEAD")).Output.Trim()

  $resultJsons = @($stagedFiles | Where-Object { $_ -like "artifacts_maestro/mainline_single_v1/*.json" -or $_ -like "artifacts_maestro/mainline_single_v1/*.jsonl" })
  $summary = [ordered]@{
    commit = $sha
    branch = [string]$manifest.branch
    result_json_files = $resultJsons
    lfs_checkpoint = $selectedRel
    staged_count = $stagedFiles.Count
  }
  $summary | ConvertTo-Json -Depth 8
} finally {
  Pop-Location
}
