@echo off
title QuantAI - Full Setup
cd /d "%~dp0"

echo ================================================
echo   QuantAI - Running full one-time setup
echo   This runs every required script, in order.
echo ================================================
echo.

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: venv not found. Run setup_windows.bat first.
    pause
    exit /b 1
)

echo.
echo [1/5] Building price database  (python pipeline.py)
echo ------------------------------------------------
python pipeline.py
if errorlevel 1 goto :fail

echo.
echo [2/5] Training RF + other core models  (python train_all_models.py)
echo ------------------------------------------------
python train_all_models.py
if errorlevel 1 goto :fail

echo.
echo [3/5] Training XGBoost models  (python train_xgboost_all.py)
echo ------------------------------------------------
python train_xgboost_all.py
if errorlevel 1 goto :fail

echo.
echo [4/5] Seeding sample trade history  (python seed_trades.py)
echo ------------------------------------------------
python seed_trades.py
if errorlevel 1 goto :fail

echo.
echo [5/5] Create your admin login  (python create_admin.py)
echo ------------------------------------------------
echo You will be asked for a username, email, and password now.
python create_admin.py
if errorlevel 1 goto :fail

echo.
echo ================================================
echo   All setup steps completed successfully!
echo.
echo   Optional extras (NOT run automatically - edit this
echo   file and remove the "REM " in front of any line
echo   below if you want them, then run this file again):
echo.
REM python train_lstm_all.py          (SeqNN model, ~10 min)
REM python train_transformer_all.py   (Transformer model, ~10 min)
REM python train_rl_agent.py          (RL Agent, ~15 min, 5 stocks only)
REM python fetch_fii_dii.py           (FII/DII institutional flow data)
echo.
echo   Next steps:
echo   1. Run start_api.bat  (or: uvicorn src.api:app --reload --port 8000)
echo   2. Open dashboard\login.html in your browser
echo ================================================
pause
exit /b 0

:fail
echo.
echo ================================================
echo   SETUP STOPPED - a step above failed.
echo   Scroll up to see which script errored, fix it,
echo   then run this file again.
echo ================================================
pause
exit /b 1
