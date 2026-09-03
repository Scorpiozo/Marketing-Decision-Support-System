@echo off
REM
REM  Bank Marketing Decision Support System - Windows launcher
REM
REM  This script:
REM    1. Creates/activates a Python virtual environment (optional)
REM    2. Installs dependencies from requirements.txt
REM    3. Runs the full training + evaluation pipeline
REM    4. Launches the Streamlit dashboard
REM

echo.
echo === Step 1: Installing dependencies ===
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Please check your Python/pip installation.
    pause
    exit /b 1
)

echo.
echo === Step 2: Training and evaluating all models (baselines + 1D CNN) ===
python train_and_evaluate.py
if errorlevel 1 (
    echo Training failed. Please check the error message above.
    pause
    exit /b 1
)

echo.
echo === Step 3: Launching the Streamlit dashboard ===
echo   (Press CTRL+C in this window to stop the dashboard)
streamlit run app\streamlit_app.py

pause
