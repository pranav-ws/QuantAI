@echo off
title QuantAI - Risk Parity Analyser
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo Risk Parity Analyser — Equal Risk Contribution Sizing
echo.
echo 1. Live mode (uses today's ensemble signals)
echo 2. Demo mode (synthetic signals, no models needed)
echo.
set /p CHOICE="Enter 1 or 2: "
if "%CHOICE%"=="2" (
    python risk_parity_analyser.py --demo
) else (
    python risk_parity_analyser.py
)
pause