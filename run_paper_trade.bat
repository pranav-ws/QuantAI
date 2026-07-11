@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python paper_trade.py >> data\log.txt 2>&1
