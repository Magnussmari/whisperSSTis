@echo off
setlocal enabledelayedexpansion

echo WhisperSST.is Launcher
echo --------------------

:: Check for Visual C++ Redistributable
echo Checking system requirements...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Visual C++ Redistributable might not be installed.
    echo Please download and install it from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

:: Run the application
echo Starting WhisperSST.is...
start "" "WhisperSST.exe"

:: Wait a moment to ensure the process starts
timeout /t 2 >nul

:: Check if the process is running
tasklist /FI "IMAGENAME eq WhisperSST.exe" 2>NUL | find /I /N "WhisperSST.exe">NUL
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start!
    echo Please check that all files are extracted correctly.
    echo.
    pause
    exit /b 1
)

exit /b 0
