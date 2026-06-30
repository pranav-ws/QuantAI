@echo off
title QuantAI - Drawdown Recovery Analyser
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo 1. Live mode (uses your real paper_trades.json)
echo 2. Demo mode (synthetic 40-trade sequence)
echo.
set /p CHOICE="Enter 1 or 2: "
if "%CHOICE%"=="2" (
    python drawdown_recovery_analyser.py --demo
) else (
    python drawdown_recovery_analyser.py
)
pause