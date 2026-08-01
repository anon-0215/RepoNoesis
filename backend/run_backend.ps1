$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if (Test-Path ".\.venv\Scripts\python.exe") {
  ".\.venv\Scripts\python.exe"
} else {
  (Get-Command python -ErrorAction Stop).Source
}

$env:PYTHONUTF8 = "1"
& $Python -m app.run_server
