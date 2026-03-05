param(
    [switch]$RebuildDataset,
    [switch]$DeleteSafeJunk,
    [ValidateSet("pilot", "full")]
    [string]$Preset = "pilot",
    [string]$DataPath = "data/all_trainable_clean_v1.jsonl",
    [string]$ManifestPath = "data/all_trainable_clean_v1.manifest.json",
    [string]$StagesDir = "data/stages_all_trainable_clean_v1",
    [string]$RunDir = "artifacts_all_trainable_clean_v1"
)

$ErrorActionPreference = "Stop"

function Test-PythonExe([string]$exePath, [bool]$RequireTorch = $false) {
    if (-not $exePath -or -not (Test-Path $exePath)) {
        return $false
    }
    try {
        & $exePath --version *> $null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        if ($RequireTorch) {
            & $exePath -c "import torch" *> $null
            if ($LASTEXITCODE -ne 0) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Resolve-PythonExe([string]$repoRoot) {
    $localVenv = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
    $candidates = @(
        $localVenv,
        "C:\\Program Files\\Python311\\python.exe"
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

Push-Location $repoRoot
try {
    if ($RebuildDataset -or -not (Test-Path $DataPath)) {
        Write-Output "==> Build clean all-data training set"
        & $pythonExe scripts/build_all_trainable_data_v1.py --out_jsonl $DataPath --manifest $ManifestPath
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    if ($DeleteSafeJunk) {
        $safeDelete = @(
            "data/chat_router_ko_v1.jsonl",
            "data/ko_chat_only_boost_v1.jsonl",
            "data/target_fixpack_v1.jsonl",
            "data/quality/_meta_clean_test_in.jsonl",
            "data/quality/_meta_clean_test_out.jsonl",
            "data/stages_50m_smoke_turbo2"
        )
        foreach ($p in $safeDelete) {
            if (Test-Path $p) {
                Remove-Item $p -Recurse -Force
            }
        }
    }

    Write-Output ("==> Train from clean all-data set ({0})" -f $Preset)
    & $PSScriptRoot\\train_rtx4060_turbo.ps1 -SourceJsonl $DataPath -StagesDir $StagesDir -RunDir $RunDir -Preset $Preset
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
