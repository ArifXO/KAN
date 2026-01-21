#!/bin/bash
# ===========================================
# Run Tabular Classification Experiment
# ===========================================
# This script runs the tabular classification experiment
# comparing KAN and MLP on the breast cancer dataset.

echo ""
echo "============================================"
echo "   KAN vs MLP: Tabular Classification"
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
python -m src.train.train_tabular --config configs/tabular.yaml

echo ""
echo "============================================"
echo "   Experiment Complete!"
echo "============================================"
echo "Check the 'results' folder for outputs."
echo ""
