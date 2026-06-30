@echo off
title QuantAI - Monte Carlo Simulation
cd /d "%~dp0"
call venv\Scripts\activate.bat
python run_monte_carlo.py
pause