# Outputs folder

Generated results produced by `train_and_evaluate.py`:

| File | Description |
|---|---|
| `model_comparison.csv` | Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC for both models on the same test set |
| `roc_curves.png` | ROC curves for Gradient Boosting and Random Forest, overlaid |
| `confusion_matrix.png` / `.csv` | Confusion matrix of the production model on the test set |
| `final_predictions.csv` | Full prediction table: `customer_record_id`, probability, lead temperature, interaction pattern, contact fatigue, recommendation for every test-set customer |
| `top_priority_customers.csv` | Top 50 customers by conversion probability |
| `run_summary.json` | Machine-readable run summary (dataset sizes, best model per metric, production model, category counts) |

These files are regenerated every time `train_and_evaluate.py` runs.
