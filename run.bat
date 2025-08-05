@echo off
echo Starting ModelBuilder...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the application
python trainer\app.py

REM Pause to see any error messages
pause