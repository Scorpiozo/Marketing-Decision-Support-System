# Outputs folder

This folder holds generated results produced by `train_and_evaluate.py`
(or by running the notebook):

| File | Description |
|---|---|
| `model_comparison.csv` | Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC for every model on the same test set |
| `cnn_training_history.png` | CNN loss / accuracy / AUC curves over training epochs |
| `cnn_confusion_matrix.png` / `.csv` | Confusion matrix for the 1D CNN on the test set |
| `roc_curves_all_models.png` | ROC curves for all models, overlaid |
| `final_predictions.csv` | Full prediction table: `customer_record_id`, probability, lead temperature, interaction pattern, contact fatigue, recommendation for every test-set customer |
| `top_priority_customers.csv` | Top 50 customers by conversion probability |
| `run_summary.json` | Small machine-readable summary of the run (dataset sizes, best model by each metric, category counts) |

These files are regenerated every time `train_and_evaluate.py` is run.
