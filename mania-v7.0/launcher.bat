@echo off
title Osumapper Launcher

echo ==================================================
echo           Osumapper UI Launcher
echo ==================================================
echo.
echo This script will set up the environment and run the application.
echo.

REM === Step 1: Check for Prerequisites ===
echo [1/4] Checking for prerequisites (Python and Node.js)...
echo.

:CHECK_PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not added to your system's PATH.
    echo Please install Python 3.8+ from python.org and ensure you check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo Python installation found.

:CHECK_NODE
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not added to your system's PATH.
    echo Please install Node.js (LTS version) from nodejs.org.
    pause
    exit /b 1
)
echo Node.js installation found.
echo.

REM === Step 2: Install Python Dependencies ===
echo [2/4] Installing Python dependencies from requirements.txt...
echo This may take several minutes, especially for TensorFlow.
echo.
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies.
    echo Please check your internet connection and pip configuration.
    pause
    exit /b 1
)
echo Python dependencies installed successfully.
echo.

REM === Step 3: Install Node.js Dependencies ===
echo [3/4] Installing Node.js dependencies from package.json...
echo.
npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies.
    echo Please check your internet connection and npm configuration.
    pause
    exit /b 1
)
echo Node.js dependencies installed successfully.
echo.

REM === Step 4: Run the Application ===
echo [4/4] Launching the Osumapper UI application...
echo A web browser window should open with the application interface.
echo Close this terminal window to stop the application.
echo.
python app.py

echo.
echo ==================================================
echo      Application has been closed.
echo ==================================================
pause