@echo off
REM ===================================================================
REM FLIGHT FARE PREDICTION - APPLICATION LAUNCHER
REM ===================================================================
REM This batch file starts the Flask web application.
REM
REM What it does:
REM 1. Displays a startup message
REM 2. Runs the main.py Flask application
REM 3. Pauses after completion to show any error messages
REM
REM Usage: Double-click this file or run from command line
REM ===================================================================

echo ============================================
echo   Flight Fare Prediction App
echo   Starting Flask Server...
echo ============================================
echo.

REM Start the Flask application
python main.py

REM Pause to keep window open if there are errors
pause
