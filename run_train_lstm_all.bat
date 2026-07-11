@echo off
title QuantAI - Train LSTM (All Nifty 50)
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo ================================================
echo   QuantAI LSTM Training
echo   This will take 10-25 minutes. Keep this window
echo   open and let it run. Do not close it.
echo ================================================
echo.
python train_lstm_all.py
pause
