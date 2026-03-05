param(
  [string]$ManifestPath = "configs/publish_manifest_mainline_v1.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $script:PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-RepoRoot {
  return (Split-Path -Parent $PSScriptRoot)
}

function Resolve-FullPath {
  param([string]$PathValue, [string]$RepoRoot)
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Pick-CandidateCheckpoint {
  param($Manifest, [string]$RepoRoot)
  $promotion = Resolve-FullPath -PathValue ([string]$Manifest.promotion_report_path) -RepoRoot $RepoRoot
  if (Test-Path -LiteralPath $promotion) {
    try {
      $payload = Get-Content -LiteralPath $promotion -Raw -Encoding utf8 | ConvertFrom-Json
      $winner = [string]$payload.winner
      if (-not [string]::IsNullOrWhiteSpace($winner)) {
        $w = Resolve-FullPath -PathValue $winner -RepoRoot $RepoRoot
        if (Test-Path -LiteralPath $w) { return $w }
      }
    } catch {}
  }

  $evalPath = Resolve-FullPath -PathValue ([string]$Manifest.eval_report_path) -RepoRoot $RepoRoot
  if (Test-Path -LiteralPath $evalPath) {
    try {
      $eval = Get-Content -LiteralPath $evalPath -Raw -Encoding utf8 | ConvertFrom-Json
      $best = $null
      $bestScore = -1.0
      foreach ($run in $eval.exact_runs) {
        $outPath = Resolve-FullPath -PathValue ([string]$run.out_json) -RepoRoot $RepoRoot
        if (-not (Test-Path -LiteralPath $outPath)) { continue }
        $res = Get-Content -LiteralPath $outPath -Raw -Encoding utf8 | ConvertFrom-Json
        foreach ($r in $res.results) {
          $score = [double]$r.exact_match
          $ck = Resolve-FullPath -PathValue ([string]$r.checkpoint) -RepoRoot $RepoRoot
          if ((Test-Path -LiteralPath $ck) -and $score -gt $bestScore) {
            $best = $ck
            $bestScore = $score
          }
        }
      }
      if ($best) { return $best }
    } catch {}
  }
  return $null
}

$repoRoot = Resolve-RepoRoot
$manifestFull = Resolve-FullPath -PathValue $ManifestPath -RepoRoot $repoRoot
if (-not (Test-Path -LiteralPath $manifestFull)) {
  throw "manifest not found: $manifestFull"
}
$manifest = Get-Content -LiteralPath $manifestFull -Raw -Encoding utf8 | ConvertFrom-Json

Push-Location $repoRoot
  try {
    $isGit = Test-Path -LiteralPath (Join-Path $repoRoot ".git")
    $branch = ""
    $origin = ""
    $lastCommit = ""
    $gitLfs = ""

    if ($isGit) {
      $b = (& git rev-parse --abbrev-ref HEAD 2>$null)
      if ($LASTEXITCODE -eq 0) { $branch = [string]$b }
      $o = (& git remote get-url origin 2>$null)
      if ($LASTEXITCODE -eq 0) { $origin = [string]$o }
      $c = (& git rev-parse HEAD 2>$null)
      if ($LASTEXITCODE -eq 0) { $lastCommit = [string]$c }
    }

    $lfs = (& git lfs version 2>$null)
    if ($LASTEXITCODE -eq 0) { $gitLfs = [string]$lfs }

  $candidate = Pick-CandidateCheckpoint -Manifest $manifest -RepoRoot $repoRoot
  $status = [ordered]@{
    repo_root = $repoRoot
    is_git_repo = $isGit
    branch = $branch
    origin = $origin
    lfs = $gitLfs
    last_commit = $lastCommit
    manifest = $manifestFull
    candidate_checkpoint = $candidate
  }
  $status | ConvertTo-Json -Depth 8
} finally {
  Pop-Location
}
