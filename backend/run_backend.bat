@echo off
setlocal

cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if not defined PYTHON_EXE (
  echo Cannot find Python.
  echo.
  echo Install Python 3.12 or create a conda environment, then run:
  echo pip install -r requirements.txt
  pause
  exit /b 1
)

"%PYTHON_EXE%" -m app.run_server
