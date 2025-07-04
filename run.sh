#!/bin/bash

# Run NIDS pipeline

echo "=== NIDS Pipeline ==="
echo

# Activate virtual environment
source venv/bin/activate

# Choose preprocessing method (grouped or enhanced)
PREPROCESSING=${1:-enhanced}

echo "Using $PREPROCESSING preprocessing"
echo

# Step 1: Preprocessing
if [ "$PREPROCESSING" = "enhanced" ]; then
    echo "[1/4] Running enhanced preprocessing..."
    python3 02_preprocessing_enhanced.py
else
    echo "[1/4] Running grouped preprocessing..."
    python3 02_preprocessing_grouped.py
fi

echo
echo "[2/5] Running feature analysis..."
python3 05_analyze_features.py

echo
echo "[3/5] Testing preprocessing results..."
python3 test_enhanced.py

echo
echo "[4/5] Training models..."
python3 03_train_improved.py

echo
echo "[5/5] Running XAI analysis..."
python3 04_xai_analysis.py

echo
echo "✓ Pipeline complete!" 