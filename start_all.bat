@echo off
setlocal

cd /d "%~dp0"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" (
  echo RepoNoesis startup failed: Windows PowerShell is unavailable.
  exit /b 1
)

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_local.ps1"
exit /b %ERRORLEVEL%

