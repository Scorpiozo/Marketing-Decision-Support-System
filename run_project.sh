#!/usr/bin/env bash
# Bank Marketing Decision Support System - macOS/Linux launcher
# 1. install dependencies  2. train both models if needed  3. start the dashboard
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1: Installing dependencies ==="
python3 -m pip install -r requirements.txt

if [ -f models/production_model.json ]; then
  echo "=== Step 2: Trained models found - skipping training (retrain: python3 train_and_evaluate.py) ==="
else
  echo "=== Step 2: Training and evaluating models (Gradient Boosting + Random Forest) ==="
  python3 train_and_evaluate.py
fi

echo "=== Step 3: Starting the dashboard: open http://localhost:5000 (CTRL+C to stop) ==="
python3 run.py
