@echo off
title QuantAI - Setup Windows Task Scheduler
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Run as Administrator for best results.
echo.
python setup_automation.py
pause
