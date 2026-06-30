@echo off
title QuantAI - Test Ensemble Model
cd /d "%~dp0"
call venv\Scripts\activate.bat
python test_ensemble.py
pause
