@echo off
title QuantAI - Train XGBoost (All Nifty 50)
cd /d "%~dp0"
call venv\Scripts\activate.bat
python train_xgboost_all.py
pause
