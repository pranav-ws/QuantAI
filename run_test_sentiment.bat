@echo off
title QuantAI - Test News Sentiment
cd /d "%~dp0"
call venv\Scripts\activate.bat
python test_sentiment.py
pause
