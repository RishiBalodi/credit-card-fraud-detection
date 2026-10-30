@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo  FinGuard: Credit Card Fraud Detection Dashboard Launcher
echo ==============================================================================
echo.

REM Verify dataset exists
set "DATASET_FOUND=0"
if exist "data\fraudTest_5000.csv" set "DATASET_FOUND=1"
if exist "data\credit_card_transactions.csv" set "DATASET_FOUND=1"

if "%DATASET_FOUND%"=="0" (
    echo [ERROR] Required dataset was not found in the 'data\' folder.
    echo Expected: data\fraudTest_5000.csv or data\credit_card_transactions.csv
    echo Please place the dataset in the data\ folder before running the dashboard.
    pause
    exit /b 1
)

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .\.venv\Scripts\activate.bat
    set "RUNNER=python"
) else if exist "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" (
    echo [INFO] Running using Python 3.11 system installation...
    set "RUNNER=C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    echo [INFO] Running using default system python...
    set "RUNNER=python"
)

REM Check if model exists, if not inform user that main.py can train it
if not exist "models\decision_tree_model" (
    echo [NOTICE] Trained model not detected at models\decision_tree_model.
    echo Running end-to-end training pipeline first (python main.py)...
    "%RUNNER%" main.py
)

echo.
echo [INFO] Starting FinGuard Streamlit Web Dashboard...
echo Local URL: http://localhost:8501
echo.
"%RUNNER%" -m streamlit run app.py
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
    exit /b 1
)
