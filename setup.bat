@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo  FinGuard: Credit Card Fraud Detection Environment Setup
echo ==============================================================================
echo.

REM Determine Python executable
set "PYTHON_EXE="

if not "%~1"=="" (
    set "PYTHON_EXE=%~1"
) else if exist "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python was not found on your system PATH or at default locations.
    echo Please install Python 3.11 or pass its full path:
    echo     setup.bat "C:\Path\To\Python311\python.exe"
    pause
    exit /b 1
)

echo [INFO] Using Python executable: %PYTHON_EXE%
"%PYTHON_EXE%" --version
if !errorlevel! neq 0 (
    echo [ERROR] Failed to execute the specified Python binary.
    pause
    exit /b 1
)

REM Verify Java installation (Required by Apache Spark)
echo.
echo [INFO] Verifying Java environment for Apache Spark...
where java >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] Java was not found on PATH. Apache Spark requires Java 17.
    echo Please install OpenJDK 17 and set JAVA_HOME.
) else (
    java -version
)

REM Step 1: Create virtual environment
echo.
echo [1/3] Creating virtual environment (.venv)...
if exist ".venv" (
    echo [INFO] Existing .venv directory found. Reusing existing environment.
) else (
    "%PYTHON_EXE%" -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created at .venv
)

REM Step 2: Upgrade pip
echo.
echo [2/3] Upgrading pip...
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo [WARNING] Pip upgrade encountered an issue. Proceeding with dependency installation...
)

REM Step 3: Install dependencies
echo.
echo [3/3] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Failed to install one or more dependencies from requirements.txt.
    echo Please inspect the console error logs above.
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo  [SUCCESS] Environment setup is complete!
echo ==============================================================================
echo To activate the virtual environment manually in PowerShell, run:
echo     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
echo     .\.venv\Scripts\Activate.ps1
echo.
echo To run the complete machine learning pipeline:
echo     python main.py
echo.
echo To launch the Streamlit fraud detection dashboard:
echo     run.bat
echo ==============================================================================
pause
