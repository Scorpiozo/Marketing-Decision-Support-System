"""
train_and_evaluate.py

End-to-end script that:
    1. Loads and preprocesses the Bank Marketing dataset (data/bank-full.csv is read, never modified)
    2. Trains the two supported models: Gradient Boosting and Random Forest
    3. Compares them on the same held-out test set
    4. Saves both models, the preprocessor, the feature order and the
       production-model selection to models/
    5. Builds the final prediction dataframe + Top Priority Customers table
       with the production model
    6. Saves outputs (metrics, plots, CSVs, run summary) to outputs/

Run from anywhere (paths are resolved relative to this file):
    python train_and_evaluate.py
    python train_and_evaluate.py --production-model random_forest
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend so this runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src import model as mdl
from src import preprocessing as prep
from src.prediction import build_prediction_dataframe, get_top_priority_customers

SELECTION_METRIC = "ROC-AUC"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the bank marketing models.")
    parser.add_argument(
        "--production-model",
        choices=["auto"] + list(mdl.MODEL_REGISTRY),
        default="auto",
        help=(
            "Model used to generate the decision-support outputs. "
            f"'auto' (default) picks the model with the highest test {SELECTION_METRIC}."
        ),
    )
    return parser.parse_args(argv)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def plot_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curves(y_test, probabilities, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    section("STEP 1: Load & preprocess data")
    data = prep.load_and_prepare(data_path=config.DATA_PATH)
    X_train_proc, X_test_proc = data["X_train_proc"], data["X_test_proc"]
    y_train, y_test = data["y_train"].values, data["y_test"].values
    prep.save_preprocessor(data["preprocessor"])
    prep.save_feature_order(data["preprocessor"])
    print("Train:", X_train_proc.shape, " Test:", X_test_proc.shape)
    print("Positive rate (train):", round(y_train.mean(), 4))

    section("STEP 2: Train models")
    results = {}
    probabilities = {}
    predictions = {}
    for key, spec in mdl.MODEL_REGISTRY.items():
        name = spec["display_name"]
        print(f"Training {name}...")
        model = spec["trainer"](X_train_proc, y_train)
        y_prob = mdl.predict_proba(model, X_test_proc)
        y_pred = (y_prob >= mdl.DECISION_THRESHOLD).astype(int)
        results[key] = mdl.evaluate_model(y_test, y_pred, y_prob, name)
        probabilities[key] = y_prob
        predictions[key] = y_pred
        print("  saved ->", mdl.save_model(model, key))

    section("STEP 3: Model comparison")
    comparison_df = pd.DataFrame(list(results.values())).round(4)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)

    best = {
        metric: comparison_df.loc[comparison_df[metric].idxmax(), "Model"]
        for metric in ["ROC-AUC", "Recall", "F1-score"]
    }
    for metric, name in best.items():
        print(f"Best model by {metric}: {name}")

    if args.production_model == "auto":
        production_key = max(results, key=lambda k: results[k][SELECTION_METRIC])
        selected_by = f"highest test {SELECTION_METRIC}"
    else:
        production_key = args.production_model
        selected_by = "manual (--production-model)"
    production_name = mdl.display_name(production_key)
    info = mdl.save_production_model_info(production_key, selected_by, results[production_key])
    print(f"Production model: {production_name} ({selected_by})")

    plot_roc_curves(
        y_test,
        {mdl.display_name(k): p for k, p in probabilities.items()},
        config.ROC_CURVES_PATH,
    )
    cm = mdl.get_confusion_matrix(y_test, predictions[production_key])
    plot_confusion_matrix(cm, production_name, config.CONFUSION_MATRIX_PNG_PATH)
    np.savetxt(config.CONFUSION_MATRIX_CSV_PATH, cm, fmt="%d", delimiter=",")

    section(f"STEP 4: Build decision-support outputs ({production_name} probabilities)")
    X_test_raw = data["X_test"].reset_index(drop=True)
    prediction_df = build_prediction_dataframe(X_test_raw, probabilities[production_key])
    prediction_df.to_csv(config.PREDICTIONS_PATH, index=False)

    top_50 = get_top_priority_customers(prediction_df, top_n=50)
    top_50.to_csv(config.TOP_PRIORITY_PATH, index=False)

    print("Lead temperature distribution:")
    print(prediction_df["lead_temperature"].value_counts())
    print("\nInteraction pattern distribution:")
    print(prediction_df["interaction_pattern"].value_counts())
    print(f"\nSaved final_predictions.csv ({len(prediction_df)} rows) and "
          f"top_priority_customers.csv (top {len(top_50)}) to {config.OUTPUTS_DIR}")

    summary = {
        "n_train": int(X_train_proc.shape[0]),
        "n_test": int(X_test_proc.shape[0]),
        "n_features_after_encoding": int(X_train_proc.shape[1]),
        "models_trained": [mdl.display_name(k) for k in mdl.MODEL_REGISTRY],
        "best_model_by_roc_auc": best["ROC-AUC"],
        "best_model_by_recall": best["Recall"],
        "best_model_by_f1": best["F1-score"],
        "production_model_used_for_final_output": production_name,
        "production_model_selected_by": selected_by,
        "production_model_test_roc_auc": info["test_metrics"]["ROC-AUC"],
        "lead_temperature_counts": prediction_df["lead_temperature"].value_counts().to_dict(),
        "interaction_pattern_counts": prediction_df["interaction_pattern"].value_counts().to_dict(),
        "contact_fatigue_counts": prediction_df["contact_fatigue"].value_counts().to_dict(),
    }
    with open(config.RUN_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll done. See the outputs/ folder for saved artifacts.")
    return summary


if __name__ == "__main__":
    main()
