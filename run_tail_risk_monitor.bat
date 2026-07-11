@echo off
title QuantAI - Tail Risk Monitor (Black Swan Detector)
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo Tail Risk Monitor — detects black swan conditions across all 50 Nifty stocks
echo.
set /p PERIOD="Lookback period (1y / 2y, default 1y): "
if "%PERIOD%"=="" set PERIOD=1y
python tail_risk_monitor.py --period %PERIOD%
pause