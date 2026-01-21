@echo off
REM ===========================================
REM Run Tabular Classification Experiment
REM ===========================================
REM This script runs the tabular classification experiment
REM comparing KAN and MLP on the breast cancer dataset.

echo.
echo ============================================
echo    KAN vs MLP: Tabular Classification
echo ============================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the training script
echo Running training script...
echo.
python -m src.train.train_tabular --config configs/tabular.yaml

echo.
echo ============================================
echo    Experiment Complete!
echo ============================================
echo Check the 'results' folder for outputs.
echo.

pause
