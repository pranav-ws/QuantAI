@echo off
title QuantAI - Setup Windows Auto-Start
echo.
echo  This will configure Windows Task Scheduler to
echo  start the QuantAI Scheduler automatically on login.
echo.
set "PROJ=%~dp0"
set "BAT=%PROJ%run_scheduler.bat"

schtasks /create /tn "QuantAI Scheduler Autostart" ^
  /tr "%BAT%" ^
  /sc ONLOGON ^
  /delay 0001:00 ^
  /f

echo.
echo  Done! QuantAI scheduler will now start automatically
echo  every time you log into Windows.
echo.
echo  To remove it:
echo    schtasks /delete /tn "QuantAI Scheduler Autostart" /f
echo.
pause
