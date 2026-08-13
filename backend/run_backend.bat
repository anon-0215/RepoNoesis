@echo off
setlocal

cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHON_EXE="
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" (
  echo Cannot find Windows PowerShell for runtime discovery.
  exit /b 1
)
for /f "usebackq delims=" %%P in (`"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\resolve_runtime.ps1" -Kind Python`) do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if not defined PYTHON_EXE (
  echo Cannot find Python.
  echo.
  echo Activate the intended Conda environment or set REPONOESIS_PYTHON.
  exit /b 1
)

"%PYTHON_EXE%" -m app.run_server
exit /b %ERRORLEVEL%
