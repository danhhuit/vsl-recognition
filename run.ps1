# VSL Recognition - Quick Start Script
# Double-click or run: powershell -ExecutionPolicy Bypass -File run.ps1

$Host.UI.RawUI.WindowTitle = "VSL Recognition"
Write-Host ""
Write-Host "  ======================================" -ForegroundColor Cyan
Write-Host "   VSL Recognition - Khoi chay nhanh" -ForegroundColor Cyan
Write-Host "  ======================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# Check Python
try {
    $pyVer = python --version 2>&1
    Write-Host "[OK] $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Khong tim thay Python 3.11+" -ForegroundColor Red
    Write-Host "        Tai tai: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Nhan Enter de thoat"
    exit 1
}

# Create venv
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Dang tao moi truong ao..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[OK] Da tao moi truong ao." -ForegroundColor Green
}

# Activate
& .venv\Scripts\Activate.ps1

# Install deps
if (-not (Test-Path ".venv\installed.flag")) {
    Write-Host "" 
    Write-Host "[INFO] Dang cai dat thu vien (lan dau mat ~5-10 phut)..." -ForegroundColor Yellow
    pip install --upgrade pip 2>$null | Out-Null
    
    Write-Host "  -> Cai dat PyTorch..." -ForegroundColor Gray
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    Write-Host "  -> Cai dat cac thu vien khac..." -ForegroundColor Gray  
    pip install -r requirements.txt
    
    New-Item -Path ".venv\installed.flag" -ItemType File -Force | Out-Null
    Write-Host "[OK] Da cai dat xong!" -ForegroundColor Green
}

# Create dirs
@("data\raw_videos", "data\keypoints", "weights", "results") | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -Path $_ -ItemType Directory -Force | Out-Null }
}

# Launch
Write-Host ""
Write-Host "  ======================================" -ForegroundColor Cyan
Write-Host "   Dang mo giao dien..." -ForegroundColor Cyan
Write-Host "   Nhan Ctrl+C de dung." -ForegroundColor Cyan
Write-Host "  ======================================" -ForegroundColor Cyan
Write-Host ""

streamlit run src/app.py --server.headless true

Read-Host "Nhan Enter de thoat"
