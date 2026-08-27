"""
train_and_evaluate.py
----------------------
End-to-end script that:
    1. Loads and preprocesses the Bank Marketing dataset
    2. Trains baseline models (Logistic Regression, Decision Tree, Random Forest)
    3. Trains the 1D CNN
    4. Compares all models on a consistent test set
    5. Saves all trained models + the preprocessor to models/
    6. Builds the final prediction dataframe + Top Priority Customers table
    7. Saves outputs (metrics, plots, CSVs) to outputs/

Run this from the project root:
    python train_and_evaluate.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend so this runs without a display
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import preprocessing as prep
from src import model as mdl
from src.prediction import build_prediction_dataframe, get_top_priority_customers

OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"


def ensure_dirs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def plot_cnn_history(history, out_path):
    hist = history.history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(hist["loss"], label="train")
    axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set_title("CNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["accuracy"], label="train")
    axes[1].plot(hist["val_accuracy"], label="validation")
    axes[1].set_title("CNN Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if "auc" in hist:
        axes[2].plot(hist["auc"], label="train")
        axes[2].plot(hist["val_auc"], label="validation")
        axes[2].set_title("CNN AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


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


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 1: Load & preprocess data")
    print("=" * 70)
    data = prep.load_and_prepare(data_path="data/bank-full.csv")
    X_train_proc, X_test_proc = data["X_train_proc"], data["X_test_proc"]
    y_train, y_test = data["y_train"].values, data["y_test"].values
    prep.save_preprocessor(data["preprocessor"], path=f"{MODELS_DIR}/preprocessor.joblib")
    prep.save_feature_order(data["preprocessor"], path=f"{MODELS_DIR}/feature_order.json")
    print("Train:", X_train_proc.shape, " Test:", X_test_proc.shape)
    print("Positive rate (train):", round(y_train.mean(), 4))

    results = []
    roc_data = {}

    # -----------------------------------------------------------------
    # STEP 2: Baseline models
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Train baseline models")
    print("=" * 70)

    print("Training Logistic Regression...")
    log_model = mdl.train_logistic_regression(X_train_proc, y_train)
    y_prob_log = log_model.predict_proba(X_test_proc)[:, 1]
    y_pred_log = (y_prob_log >= 0.5).astype(int)
    results.append(mdl.evaluate_model(y_test, y_pred_log, y_prob_log, "Logistic Regression"))
    mdl.save_sklearn_model(log_model, f"{MODELS_DIR}/logistic_regression.joblib")

    print("Training Decision Tree (balanced)...")
    dt_model = mdl.train_decision_tree(X_train_proc, y_train, balanced=True)
    y_prob_dt = dt_model.predict_proba(X_test_proc)[:, 1]
    y_pred_dt = (y_prob_dt >= 0.5).astype(int)
    results.append(mdl.evaluate_model(y_test, y_pred_dt, y_prob_dt, "Decision Tree (balanced)"))
    mdl.save_sklearn_model(dt_model, f"{MODELS_DIR}/decision_tree.joblib")

    print("Training Random Forest...")
    rf_model = mdl.train_random_forest(X_train_proc, y_train)
    y_prob_rf = rf_model.predict_proba(X_test_proc)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    results.append(mdl.evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest"))
    mdl.save_sklearn_model(rf_model, f"{MODELS_DIR}/random_forest.joblib")

    # -----------------------------------------------------------------
    # STEP 3: 1D CNN
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Train 1D CNN")
    print("=" * 70)

    cnn_model, history = mdl.train_cnn_model(
        X_train_proc, y_train, epochs=50, batch_size=256, verbose=2
    )
    y_prob_cnn = mdl.cnn_predict_proba(cnn_model, X_test_proc)
    y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
    results.append(mdl.evaluate_model(y_test, y_pred_cnn, y_prob_cnn, "1D CNN"))
    mdl.save_cnn_model(cnn_model, f"{MODELS_DIR}/cnn_model.keras")

    plot_cnn_history(history, f"{OUTPUTS_DIR}/cnn_training_history.png")

    # -----------------------------------------------------------------
    # STEP 4: Model comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Model comparison")
    print("=" * 70)
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.round(4)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f"{OUTPUTS_DIR}/model_comparison.csv", index=False)

    # Confusion matrix for the CNN (the model used downstream for the DSS)
    cm = mdl.get_confusion_matrix(y_test, y_pred_cnn)
    plot_confusion_matrix(cm, "1D CNN", f"{OUTPUTS_DIR}/cnn_confusion_matrix.png")
    np.savetxt(f"{OUTPUTS_DIR}/cnn_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # ROC curves for all models on one plot
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, y_prob in [
        ("Logistic Regression", y_prob_log),
        ("Decision Tree", y_prob_dt),
        ("Random Forest", y_prob_rf),
        ("1D CNN", y_prob_cnn),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/roc_curves_all_models.png", dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------
    # STEP 5: Final Marketing Decision Support System outputs
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Build final prediction dataframe (using CNN probabilities)")
    print("=" * 70)
    best_by_roc_auc = comparison_df.loc[comparison_df["ROC-AUC"].idxmax(), "Model"]
    best_by_recall = comparison_df.loc[comparison_df["Recall"].idxmax(), "Model"]
    best_by_f1 = comparison_df.loc[comparison_df["F1-score"].idxmax(), "Model"]
    print(f"Best model by ROC-AUC:  {best_by_roc_auc}")
    print(f"Best model by Recall:   {best_by_recall}")
    print(f"Best model by F1-score: {best_by_f1}")
    print(
        "NOTE: The table above is the honest, unedited comparison for this "
        "run - whichever model tops each metric is reported as-is, and it "
        "will not always be the CNN. The 1D CNN is still used to generate "
        "the final decision-support outputs below because it is the "
        "project's required advanced deep-learning model, not because it "
        "is guaranteed to be the top performer on every metric. See "
        "README.md section 8 for the full discussion of this trade-off."
    )

    X_test_raw = data["X_test"].reset_index(drop=True)
    prediction_df = build_prediction_dataframe(X_test_raw, y_prob_cnn)
    prediction_df.to_csv(f"{OUTPUTS_DIR}/final_predictions.csv", index=False)

    top_50 = get_top_priority_customers(prediction_df, top_n=50)
    top_50.to_csv(f"{OUTPUTS_DIR}/top_priority_customers.csv", index=False)

    print("Lead temperature distribution:")
    print(prediction_df["lead_temperature"].value_counts())
    print("\nInteraction pattern distribution:")
    print(prediction_df["interaction_pattern"].value_counts())

    print(f"\nSaved final_predictions.csv ({len(prediction_df)} rows) and "
          f"top_priority_customers.csv (top {len(top_50)}) to {OUTPUTS_DIR}/")

    # Save a small run summary
    summary = {
        "n_train": int(X_train_proc.shape[0]),
        "n_test": int(X_test_proc.shape[0]),
        "n_features_after_encoding": int(X_train_proc.shape[1]),
        "best_model_by_roc_auc": best_by_roc_auc,
        "best_model_by_recall": best_by_recall,
        "best_model_by_f1": best_by_f1,
        "production_model_used_for_final_output": "1D CNN",
        "cnn_test_roc_auc": float(results[-1]["ROC-AUC"]),
        "lead_temperature_counts": prediction_df["lead_temperature"].value_counts().to_dict(),
        "interaction_pattern_counts": prediction_df["interaction_pattern"].value_counts().to_dict(),
        "contact_fatigue_counts": prediction_df["contact_fatigue"].value_counts().to_dict(),
    }
    with open(f"{OUTPUTS_DIR}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll done. See the outputs/ folder for saved artifacts.")


if __name__ == "__main__":
    main()
