@echo off
title QuantAI - Train All Models
cd /d "%~dp0"
call venv\Scripts\activate.bat
python train_all_models.py
pause
