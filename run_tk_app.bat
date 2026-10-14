@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe tk_app.py
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Den brethike to python sto PATH.
    echo Katevase to apo https://www.python.org/downloads/ kai tikare "Add python.exe to PATH".
    pause
    exit /b 1
  )
  python tk_app.py
)
if errorlevel 1 (
  echo.
  echo [ERROR] To tk_app.py ekleise me sfalma.
  pause
)
