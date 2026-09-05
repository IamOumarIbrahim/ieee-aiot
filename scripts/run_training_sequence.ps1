<#
.SYNOPSIS
    run_training_sequence.ps1 - Automated PowerShell Training Sequence Runner for IEEE AIoT Benchmark.

.DESCRIPTION
    Executes the frozen experimental training protocol for edge driver monitoring on NVIDIA RTX 4060 (8 GB VRAM).
    Supports YOLO11n, YOLO26n, and D-FINE-N across all 5 negative-frame ratios (0%, 20%, 40%, 60%, 80%).

.PARAMETER Detector
    Target detector architecture: 'yolo11n', 'yolo26n', 'dfine', or 'all'. Default is 'yolo11n'.

.PARAMETER Splits
    Comma-separated list of split identifiers to train (e.g., '00,20' or 'all'). Default is 'all'.

.PARAMETER Epochs
    Number of training epochs. Default is 0 (uses frozen protocol: 100 for YOLO, 160 for D-FINE).

.PARAMETER Batch
    Physical batch size. Default is 0 (uses frozen protocol: 16 for YOLO, 4 for D-FINE).

.PARAMETER Device
    CUDA visible device index (e.g., '0' or 'cpu'). Default is '0'.

.PARAMETER DryRun
    When specified, validates dataset configs, manifest paths, and checkpoints without training.

.PARAMETER SkipVerification
    When specified, skips running 'src/data/verify_splits.py' prior to training.

.EXAMPLE
    .\scripts\run_training_sequence.ps1 -Detector yolo11n -DryRun

.EXAMPLE
    .\scripts\run_training_sequence.ps1 -Detector yolo11n

.EXAMPLE
    .\scripts\run_training_sequence.ps1 -Detector yolo26n -Splits "00,20"

.EXAMPLE
    .\scripts\run_training_sequence.ps1 -Detector all
#>

[CmdletBinding()]
param(
    [ValidateSet("yolo11n", "yolo26n", "dfine", "all")]
    [string]$Detector = "yolo11n",

    [string]$Splits = "all",

    [int]$Epochs = 0,

    [int]$Batch = 0,

    [string]$Device = "0",

    [switch]$DryRun,

    [switch]$Amp,

    [switch]$SkipVerification
)

$ErrorActionPreference = "Continue"
$env:PYTHONWARNINGS = "ignore"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path

# Guard: Skip redundant single-split test (#5) per user request
if ($Splits -eq "20" -or $Splits -eq "train_20_low_neg") {
    Write-Host "`n[SKIP] Single-split test (-Splits '$Splits') skipped per user configuration.`n" -ForegroundColor Cyan
    exit 0
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  IEEE AIoT Driver Monitoring - Training Sequence Runner   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Repository Root : $RepoRoot"
Write-Host "  Target Detector : $Detector"
Write-Host "  Target Splits   : $Splits"
Write-Host "  Device          : $Device"
Write-Host "  Dry-Run Mode    : $DryRun"
Write-Host "============================================================`n"

# 1. Environment & CUDA Verification
Write-Host "[1/3] Checking Python & CUDA Environment..." -ForegroundColor Yellow
$PythonCheck = & python "$RepoRoot\scripts\check_env.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to execute Python environment check: $PythonCheck"
    exit 1
}
Write-Host "  $PythonCheck" -ForegroundColor Green

# 2. Dataset Split Verification
if (-not $SkipVerification) {
    Write-Host "`n[2/3] Verifying Dataset Splits & Configuration Integrity..." -ForegroundColor Yellow
    & python "$RepoRoot\src\data\verify_splits.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Split verification failed with exit code $LASTEXITCODE."
        exit 1
    }
    Write-Host "  Split and configuration integrity verified successfully." -ForegroundColor Green
} else {
    Write-Host "`n[2/3] Skipping dataset split verification (-SkipVerification specified)." -ForegroundColor Gray
}

# 3. Launch Training Sequence
Write-Host "`n[3/3] Initiating Detector Training Sequence..." -ForegroundColor Yellow

$DetectorsToRun = @()
if ($Detector -eq "all") {
    $DetectorsToRun = @("yolo11n", "yolo26n", "dfine")
} else {
    $DetectorsToRun = @($Detector)
}

foreach ($Det in $DetectorsToRun) {
    Write-Host "`n>>> Processing Detector: $Det <<<" -ForegroundColor Magenta

    if ($Det -in @("yolo11n", "yolo26n")) {
        $ModelFile = "$RepoRoot\$Det.pt"
        if (-not (Test-Path $ModelFile)) {
            Write-Host "  [NOTICE] Local weights file not found at $ModelFile. Ultralytics will auto-download $Det.pt if available." -ForegroundColor Yellow
            $ModelFile = "$Det.pt"
        }

        $ActualEpochs = if ($Epochs -gt 0) { $Epochs } else { 100 }
        $ActualBatch = if ($Batch -gt 0) { $Batch } else { 16 }

        $YoloArgs = @(
            "$RepoRoot\src\training\train_yolo_sweep.py",
            "--model", $ModelFile,
            "--epochs", $ActualEpochs,
            "--batch", $ActualBatch,
            "--device", $Device,
            "--splits", $Splits
        )

        if ($DryRun) {
            $YoloArgs += "--dry-run"
        }

        if ($Amp) {
            $YoloArgs += "--amp"
        }

        Write-Host "  Command: python $($YoloArgs -join ' ')" -ForegroundColor Gray
        & python @YoloArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training sweep for $Det failed with exit code $LASTEXITCODE."
            exit $LASTEXITCODE
        }
        Write-Host "  [OK] Training sweep for $Det completed successfully." -ForegroundColor Green

    } elseif ($Det -eq "dfine") {
        $DfineDir = "$RepoRoot\DFINE"
        if (-not (Test-Path "$DfineDir\train.py")) {
            Write-Host "  [NOTICE] Upstream D-FINE engine not found at $DfineDir. Cloning Peterande/D-FINE..." -ForegroundColor Yellow
            git clone https://github.com/Peterande/D-FINE.git $DfineDir
        }
        $DfineArgs = @(
            "$RepoRoot\src\training\train_dfine_sweep.py",
            "--dfine-dir", $DfineDir,
            "--device", $Device,
            "--splits", $Splits
        )

        if ($DryRun) {
            $DfineArgs += "--dry-run"
        }

        Write-Host "  Command: python $($DfineArgs -join ' ')" -ForegroundColor Gray
        & python @DfineArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training sweep for D-FINE-N failed with exit code $LASTEXITCODE."
            exit $LASTEXITCODE
        }
        Write-Host "  [OK] Training sweep for D-FINE-N completed successfully." -ForegroundColor Green
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  All Requested Sequences Completed Successfully!          " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Check output tables and metrics under: $RepoRoot\runs\"
