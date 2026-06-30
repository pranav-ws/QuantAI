@echo off
title QuantAI - Correlation Matrix & Diversification View
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo Choose lookback window:
echo   1  = 1 year
echo   2  = 2 years  (default)
echo   3  = 3 years
echo   5  = 5 years
echo.
set /p CHOICE="Enter 1 / 2 / 3 / 5 (or press Enter for 2y): "
if "%CHOICE%"=="1" set PERIOD=1y
if "%CHOICE%"=="3" set PERIOD=3y
if "%CHOICE%"=="5" set PERIOD=5y
if not defined PERIOD set PERIOD=2y
echo Running correlation matrix for %PERIOD% window...
python correlation_matrix.py --period %PERIOD%
pause