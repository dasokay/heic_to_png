@echo off
REM Starts the HEIC -> PNG converter using the project's virtual environment.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
echo.
echo Open http://127.0.0.1:5000 in your browser.
echo Press Ctrl+C to stop the server.
echo.
.venv\Scripts\python.exe app.py
