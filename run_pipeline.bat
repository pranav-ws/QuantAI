@echo off
title QuantAI - Data Pipeline
cd /d "%~dp0"
call venv\Scripts\activate.bat
python pipeline.py
pause
