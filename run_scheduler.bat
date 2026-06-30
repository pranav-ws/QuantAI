@echo off
title QuantAI - Daily Scheduler (keep this window open)
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo  ================================================================
echo   QuantAI Scheduler  --  Daily Automation Engine
echo  ================================================================
echo   Jobs:
echo     06:00 AM  Data refresh (pipeline.py)
echo     09:00 AM  Morning Telegram briefing
echo     03:45 PM  Post-market scan (paper_trade.py)
echo     04:00 PM  Evening signal alert (Telegram)
echo     05:00 PM  Weekly summary (Fridays only)
echo  ================================================================
echo   Minimise this window. DO NOT close it.
echo   Output is also saved to: data\scheduler.log
echo  ================================================================
echo.
python scheduler.py
pause
