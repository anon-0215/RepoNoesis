@echo off
cd /d "%~dp0"
if not exist "%~dp0app\runtime\python.exe" (
  echo Please fully extract the ZIP and keep the app folder beside this file.
  pause
  exit /b 1
)
"%~dp0app\runtime\python.exe" -I -B "%~dp0app\desktop\desktop_host.py" %*
if errorlevel 1 pause
