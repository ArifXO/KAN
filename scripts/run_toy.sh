#!/bin/bash
# ===========================================
# Run Toy Function Regression Experiment
# ===========================================
# This script runs the toy function regression experiment
# comparing KAN and MLP on a synthetic 2D function.

echo ""
echo "============================================"
echo "   KAN vs MLP: Toy Function Regression"
echo "============================================"
echo ""

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the training script
echo "Running training script..."
echo ""
python -m src.train.train_toy --config configs/toy.yaml

echo ""
echo "============================================"
echo "   Experiment Complete!"
echo "============================================"
echo "Check the 'results' folder for outputs."
echo ""
