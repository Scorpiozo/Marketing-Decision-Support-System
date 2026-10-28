@echo off
REM
REM  Bank Marketing Decision Support System - Windows launcher
REM
REM  1. Installs dependencies from requirements.txt
REM  2. Trains Gradient Boosting + Random Forest if no trained models exist
REM  3. Starts the dashboard at http://localhost:5000
REM
cd /d "%~dp0"

echo.
echo === Step 1: Installing dependencies ===
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Please check your Python/pip installation.
    pause
    exit /b 1
)

echo.
if exist models\production_model.json (
    echo === Step 2: Trained models found - skipping training ===
    echo   To retrain: python train_and_evaluate.py
) else (
    echo === Step 2: Training and evaluating models (Gradient Boosting + Random Forest) ===
    python train_and_evaluate.py
    if errorlevel 1 (
        echo Training failed. Please check the error message above.
        pause
        exit /b 1
    )
)

echo.
echo === Step 3: Starting the dashboard ===
echo   Open http://localhost:5000 in your browser (CTRL+C here to stop)
start "" http://localhost:5000
python run.py

pause
