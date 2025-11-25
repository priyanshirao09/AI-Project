@echo off
:: Ensure we are in the script's directory
cd /d "%~dp0"

echo ==================================================
echo          SQL Injection Detection System
echo ==================================================
echo.

:: 1. Check if Python is reachable
python --version >NUL 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python and check "Add Python to PATH" during installation.
    pause
    exit /b
)

:: 2. Check if Streamlit is installed
python -c "import streamlit" 2>NUL
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit library not found.
    echo Installing Streamlit now...
    pip install streamlit
    echo.
)

:: 3. Run the App using the Python Module method (More reliable)
echo Launching Interface...
echo (You can close this window to stop the server)
echo.

:: This method works even if 'streamlit' isn't in your system PATH
python -m streamlit run app.py

pause