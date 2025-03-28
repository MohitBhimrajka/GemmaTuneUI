@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo    GemmaTuneUI - Easy Gemma Fine-Tuning Setup     
echo =====================================================
echo.

:: Check if Python is installed
echo Checking Python version...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.9 or newer.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if !PYTHON_MAJOR! LSS 3 (
    echo Error: Python 3.9 or newer is required (found !PYTHON_VERSION!)
    echo Please install a newer version of Python and try again.
    pause
    exit /b 1
)

if !PYTHON_MAJOR! EQU 3 (
    if !PYTHON_MINOR! LSS 9 (
        echo Error: Python 3.9 or newer is required (found !PYTHON_VERSION!)
        echo Please install a newer version of Python and try again.
        pause
        exit /b 1
    )
)

echo Python !PYTHON_VERSION! detected

:: Check for NVIDIA GPU
echo Checking for NVIDIA GPU and CUDA...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: NVIDIA GPU not detected or nvidia-smi not in PATH
    echo WARNING: This application requires an NVIDIA GPU with CUDA support for fine-tuning
    echo WARNING: You can still continue, but fine-tuning will fail without GPU acceleration
    
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i "!CONTINUE!" neq "y" (
        exit /b 1
    )
) else (
    echo NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

:: Set up virtual environment
set VENV_DIR=venv

echo Setting up virtual environment...
if not exist %VENV_DIR% (
    echo Creating new virtual environment in .\%VENV_DIR%
    python -m venv %VENV_DIR%
    set FRESH_INSTALL=1
) else (
    echo Using existing virtual environment in .\%VENV_DIR%
    set FRESH_INSTALL=0
)

:: Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

:: Install or update dependencies
if %FRESH_INSTALL% equ 1 (
    echo Installing dependencies (this may take a while)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Checking for missing dependencies...
    pip install -r requirements.txt
)

echo Dependencies installed

:: Run the app
echo Starting GemmaTuneUI...
echo The application will open in your web browser
streamlit run app.py

:: Deactivate virtual environment when done
call %VENV_DIR%\Scripts\deactivate.bat

endlocal 