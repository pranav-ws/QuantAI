@echo off
title QuantAI - Setup
cd /d "%~dp0"

echo ================================================
echo   QuantAI - Windows Setup
echo ================================================
echo.

echo [1/3] Creating virtual environment (venv)...
python -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found. Install Python 3.10-3.13 from
    echo https://www.python.org/downloads/ and make sure "Add Python to PATH"
    echo is checked during installation, then run this file again.
    pause
    exit /b 1
)

echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/3] Installing required packages (this can take a few minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================================
echo   Setup complete!
echo   Next: run run_pipeline.bat to build the database
echo ================================================
pause
