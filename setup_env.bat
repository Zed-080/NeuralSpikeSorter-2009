@echo off
echo ==========================================
echo Setting up Neural Spike Sorter Environment
echo ==========================================

:: 1. Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    pause
    exit /b
)

:: 2. Create Virtual Environment if it doesn't exist
if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
) else (
    echo [INFO] Virtual environment already exists.
)

:: 3. Activate and Install
echo [INFO] Installing requirements...
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ==========================================
echo Setup Complete!
echo To run the pipeline, use:
echo .venv\Scripts\python main_runner.py
echo ==========================================
pause