@echo off
title QuantAI - Train RL Agent
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Training RL Agent on 5 stocks (300 episodes each)...
python train_rl_agent.py
pause
