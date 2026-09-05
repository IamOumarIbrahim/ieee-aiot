<#
.SYNOPSIS
    run_rq2_and_populate_tables.ps1 - Automated Post-RQ1 Pipeline Runner.
.DESCRIPTION
    Runs RQ2 hard-negative mining, trains curated models, calculates nuisance alerts,
    populates Tables III, IV, and V in docs/manuscript/main.tex, and recompiles main.pdf.
#>
[CmdletBinding()]
param()

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  IEEE AIoT 2026 - Post-RQ1 Auto Runner & Table Populator   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& python "$RepoRoot\scripts\run_rq2_and_populate_tables.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Write-Host "`n[SUCCESS] All benchmark tasks finished and manuscript PDF updated!" -ForegroundColor Green
