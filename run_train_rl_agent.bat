@echo off
title QuantAI - RL Agent Training
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo  QuantAI - Training RL Agents (all 50 Nifty stocks)
echo  This will take 3-5 minutes. Watch the progress bars!
echo.
python train_rl_agent.py
echo.
pause
