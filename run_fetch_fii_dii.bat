@echo off
title QuantAI - Fetch FII/DII Data
cd /d "%~dp0"
call venv\Scripts\activate.bat
python fetch_fii_dii.py
pause
