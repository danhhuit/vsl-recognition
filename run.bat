@echo off
chcp 65001 >nul
title VSL Recognition - Khởi chạy
echo.
echo  ======================================
echo   VSL Recognition - Khởi chạy nhanh
echo  ======================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Khong tim thay Python. Vui long cai dat Python 3.11+
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create venv if not exists
if not exist ".venv" (
    echo [INFO] Dang tao moi truong ao...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Khong the tao moi truong ao.
        pause
        exit /b 1
    )
    echo [OK] Da tao moi truong ao.
    echo.
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install dependencies if needed
if not exist ".venv\installed.flag" (
    echo [INFO] Dang cai dat thu vien...
    echo        (Lan dau co the mat 5-10 phut)
    echo.
    pip install --upgrade pip >nul 2>&1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if %errorlevel% neq 0 (
        echo [ERROR] Khong the cai dat PyTorch.
        pause
        exit /b 1
    )
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Khong the cai dat thu vien.
        pause
        exit /b 1
    )
    echo. > .venv\installed.flag
    echo [OK] Da cai dat xong thu vien.
    echo.
)

:: Create data dirs
if not exist "data\raw_videos" mkdir "data\raw_videos"
if not exist "data\keypoints" mkdir "data\keypoints"
if not exist "weights" mkdir "weights"
if not exist "results" mkdir "results"

:: Run Streamlit
echo.
echo  ======================================
echo   Dang mo giao dien web...
echo   Nhan Ctrl+C de dung.
echo  ======================================
echo.
streamlit run src/app.py --server.headless true

pause
