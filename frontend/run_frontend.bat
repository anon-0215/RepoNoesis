@echo off
setlocal

cd /d "%~dp0"
set "NPM_CMD="
for /f "delims=" %%N in ('where npm 2^>nul') do if not defined NPM_CMD set "NPM_CMD=%%N"

if not defined NPM_CMD (
  echo Cannot find npm.
  echo.
  echo Install Node.js, then run this script again.
  pause
  exit /b 1
)

if not exist "node_modules" (
  "%NPM_CMD%" ci
)

"%NPM_CMD%" run dev
