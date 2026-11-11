@echo off
cd /d "%~dp0"
"%~dp0runtime\python.exe" -I -B "%~dp0desktop\desktop_host.py" %*
if errorlevel 1 pause
