#!/usr/bin/env pwsh
# Docker startup script for TB Drug Discovery
# Optimized for 16GB RAM / 4C8T system

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("api", "train", "jupyter", "mlflow", "full", "down")]
    [string]$Mode,
    
    [switch]$Build,
    [switch]$NoCache
)

$ComposeFile = "docker-compose.optimal.yml"

function Show-Resources {
    Write-Host "`n=== System Resources ===" -ForegroundColor Cyan
    $mem = Get-CimInstance Win32_ComputerSystem
    $totalRAM = [math]::Round($mem.TotalPhysicalMemory / 1GB, 2)
    Write-Host "Total RAM: $totalRAM GB" -ForegroundColor White
    
    $dockerInfo = docker system info --format "Docker Memory: {{.MemTotal}}" 2>$null
    if ($dockerInfo) {
        Write-Host "Docker Configured: $dockerInfo" -ForegroundColor White
    }
    
    $running = docker ps -q 2>$null | Measure-Object | Select-Object -ExpandProperty Count
    Write-Host "Running containers: $running" -ForegroundColor White
    Write-Host "========================`n" -ForegroundColor Cyan
}

function Start-Api {
    Write-Host "Starting API mode (2GB RAM)..." -ForegroundColor Green
    if ($Build) {
        docker-compose -f $ComposeFile build api
    }
    docker-compose -f $ComposeFile up -d api
    Write-Host "API available at: http://localhost:8000" -ForegroundColor Yellow
    Write-Host "Health check: http://localhost:8000/health" -ForegroundColor Yellow
}

function Start-Train {
    Write-Host "Starting Training mode (6GB RAM)..." -ForegroundColor Green
    if ($Build) {
        docker-compose -f $ComposeFile build trainer
    }
    docker-compose -f $ComposeFile up -d trainer
    Write-Host "Training container ready. Enter with:" -ForegroundColor Yellow
    Write-Host "  docker exec -it tb-drug-trainer bash" -ForegroundColor Cyan
    Write-Host "`nExample training:" -ForegroundColor Yellow
    Write-Host "  docker exec tb-drug-trainer python scripts/train_qsar.py" -ForegroundColor Cyan
    Write-Host "  docker exec tb-drug-trainer python scripts/train_vae.py --epochs 10 --batch-size 32" -ForegroundColor Cyan
}

function Start-Jupyter {
    Write-Host "Starting Jupyter mode (3GB RAM)..." -ForegroundColor Green
    if ($Build) {
        docker-compose -f $ComposeFile build jupyter
    }
    docker-compose -f $ComposeFile up -d jupyter
    Write-Host "Jupyter Lab available at: http://localhost:8888" -ForegroundColor Yellow
    Write-Host "No token required (development mode)" -ForegroundColor Yellow
}

function Start-Full {
    Write-Host "Starting FULL stack (API + Jupyter + MLflow = ~5.5GB)..." -ForegroundColor Green
    Write-Host "WARNING: Don't start trainer in this mode!" -ForegroundColor Red
    if ($Build) {
        docker-compose -f $ComposeFile build
    }
    docker-compose -f $ComposeFile up -d api jupyter mlflow
    Write-Host "`nAll services:" -ForegroundColor Yellow
    Write-Host "  API:     http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  Jupyter: http://localhost:8888" -ForegroundColor Cyan
    Write-Host "  MLflow:  http://localhost:5000" -ForegroundColor Cyan
}

function Stop-All {
    Write-Host "Stopping all containers..." -ForegroundColor Red
    docker-compose -f $ComposeFile down
    Write-Host "Done!" -ForegroundColor Green
}

# Main
Show-Resources

switch ($Mode) {
    "api" { Start-Api }
    "train" { Start-Train }
    "jupyter" { Start-Jupyter }
    "full" { Start-Full }
    "down" { Stop-All }
    default { 
        Write-Host "Usage: .\docker-start.ps1 -Mode [api|train|jupyter|full|down] [-Build]" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nMonitor resources: docker stats" -ForegroundColor Magenta
Write-Host "View logs: docker logs -f <container_name>" -ForegroundColor Magenta
