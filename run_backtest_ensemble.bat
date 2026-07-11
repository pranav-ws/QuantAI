@echo off
title QuantAI - Ensemble Backtest
cd /d "%~dp0"
call venv\Scripts\activate.bat
python backtest_ensemble.py
pause
