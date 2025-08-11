@echo off
cls
color 0A
echo.
echo ======================================================================
echo.
echo    ####    B A S E M E N T   B R E W   A I
echo    #  #    +-----------------------------+
echo    ####    ^| Industrial ML Terminal v1.0 ^|
echo    #  #    +-----------------------------+
echo    ####              
echo                     Version 1.0
echo.
echo ======================================================================
echo.
echo [SYSTEM] Initializing basement brew environment...
echo.

REM Try to use venv if it exists, otherwise use global Python
if exist "venv\Scripts\python.exe" (
    echo [SYSTEM] Using virtual environment Python
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    echo [SYSTEM] Using global Python installation
    set PYTHON_EXE=python
)

echo [SYSTEM] GPU Status: 
nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader 2>nul || echo [WARNING] No NVIDIA GPU detected
echo.
echo [SYSTEM] Starting BasementBrewAI terminal...
echo.

REM Run the application
%PYTHON_EXE% trainer\app.py

REM Pause to see any error messages
echo.
echo [SYSTEM] BasementBrewAI terminated.
pause