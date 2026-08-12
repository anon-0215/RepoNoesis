@echo off
setlocal

cd /d "%~dp0"
set "NODE_EXE="
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" (
  echo Cannot find Windows PowerShell for runtime discovery.
  exit /b 1
)
for /f "usebackq delims=" %%N in (`"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\resolve_runtime.ps1" -Kind Node`) do if not defined NODE_EXE set "NODE_EXE=%%N"

if not defined NODE_EXE (
  echo Cannot find Node.js.
  echo.
  echo Install Node.js or set REPONOESIS_NODE.
  exit /b 1
)

if not exist "node_modules\vite\bin\vite.js" (
  echo Frontend dependencies are missing. Install the declared dependencies first.
  exit /b 1
)

"%NODE_EXE%" node_modules\vite\bin\vite.js --configLoader native --host 127.0.0.1 --port 5173
exit /b %ERRORLEVEL%
