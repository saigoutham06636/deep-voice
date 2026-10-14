@echo off
echo ===================================================
echo   DeepFake Voice Phishing Detection System Setup
echo ===================================================
echo.

echo [1/3] Creating virtual environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment. Ensure Python is installed and added to PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo ===================================================
echo   Setup completed successfully!
echo   Run run.bat to start the system.
echo ===================================================
echo.
pause
