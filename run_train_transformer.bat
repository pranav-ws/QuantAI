@echo off
title QuantAI - Train Transformer (All 50 Stocks)
cd /d "%~dp0"
call venv\Scripts\activate.bat
python train_transformer_all.py
pause
