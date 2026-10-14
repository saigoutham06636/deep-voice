@echo off
echo ===================================================
echo   DeepFake Voice Phishing Detection System Launcher
echo ===================================================
echo.

if not exist venv (
    echo [WARNING] Virtual environment 'venv' not found.
    echo Running setup.bat first...
    call setup.bat
)

echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

echo.
echo Starting Flask server...
python -m backend.app
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Backend application terminated with exit code %errorlevel%.
    pause
    exit /b %errorlevel%
)
