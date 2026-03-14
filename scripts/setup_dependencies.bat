@echo off
setlocal enabledelayedexpansion

echo Setting up dependencies for Windows...

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or later from python.org
    pause
    exit /b 1
)

:: Check if pip is installed
pip --version > nul 2>&1
if errorlevel 1 (
    echo pip is not installed! Please install pip with Python
    pause
    exit /b 1
)

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

:: Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

:: Check for Visual C++ Redistributable
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0" > nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Visual C++ Redistributable might not be installed.
    echo Please download and install it from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
)

echo.
echo Dependencies installed successfully!
echo You can now run the WhisperSST application.
echo.
pause
