@echo off
REM ===================================================================
REM FLIGHT FARE PREDICTION - DEPENDENCY INSTALLER
REM ===================================================================
REM This batch file installs all required Python libraries.
REM
REM What it does:
REM 1. Uses pip to install packages from requirements.txt
REM 2. Displays status messages
REM 3. Pauses at the end to show completion message
REM
REM Prerequisites:
REM - Python must be installed
REM - Python must be added to system PATH
REM
REM Packages installed:
REM - pandas: Data manipulation
REM - numpy: Numerical computing
REM - scikit-learn: Machine learning
REM - flask: Web framework
REM - flask-cors: CORS support
REM - openpyxl: Excel file reading
REM - matplotlib: Plotting (optional)
REM - seaborn: Statistical visualization (optional)
REM
REM Usage: Double-click this file before first run
REM ===================================================================

echo ============================================
echo   Flight Fare Prediction
echo   Installing Dependencies...
echo ============================================
echo.

REM Install all packages from requirements.txt
python -m pip install -r requirements.txt

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
pause
