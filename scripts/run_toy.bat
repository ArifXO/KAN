@echo off
REM ===========================================
REM Run Toy Function Regression Experiment
REM ===========================================
REM This script runs the toy function regression experiment
REM comparing KAN and MLP on a synthetic 2D function.

echo.
echo ============================================
echo    KAN vs MLP: Toy Function Regression
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
python -m src.train.train_toy --config configs/toy.yaml

echo.
echo ============================================
echo    Experiment Complete!
echo ============================================
echo Check the 'results' folder for outputs.
echo.

pause
