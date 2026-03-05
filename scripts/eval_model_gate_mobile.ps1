param(
    [string]$Checkpoint = "artifacts_slm_mit_unified_v4/slm_best.pt",
    [switch]$UseEMA,
    [switch]$NoEMA,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cpu"
)

$argsList = @(
    "scripts/eval_model_gate_mobile.py",
    "--checkpoint", $Checkpoint,
    "--device", $Device
)

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

python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Mobile gate evaluation completed. cpu_threads=2, perf_cap=20%"
