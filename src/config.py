"""
Central project paths.

All paths are absolute and anchored at the project root, so the training
script, the dashboard and the tests work regardless of the current working
directory they are launched from.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATA_PATH = os.path.join(DATA_DIR, "bank-full.csv")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
FEATURE_ORDER_PATH = os.path.join(MODELS_DIR, "feature_order.json")
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "production_model.json")

PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "final_predictions.csv")
TOP_PRIORITY_PATH = os.path.join(OUTPUTS_DIR, "top_priority_customers.csv")
MODEL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
ROC_CURVES_PATH = os.path.join(OUTPUTS_DIR, "roc_curves.png")
CONFUSION_MATRIX_PNG_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
CONFUSION_MATRIX_CSV_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.csv")
RUN_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "run_summary.json")
