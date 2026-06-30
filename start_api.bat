@echo off
title QuantAI - API Server
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting QuantAI API on http://127.0.0.1:8000
echo Keep this window open while using the dashboard.
echo Press CTRL+C to stop the server.
echo.
uvicorn src.api:app --reload --port 8000
pause
