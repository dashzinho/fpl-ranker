@echo off
cd /d "%~dp0"
echo Starting FPL Rankings (close this window to stop)...
python fpl.py --serve --open
pause
