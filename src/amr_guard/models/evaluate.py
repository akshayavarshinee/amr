"""
Model evaluation utilities.

Returns a metrics dict with accuracy, precision, recall, F1, ROC-AUC,
PR-AUC, MCC, and saves a confusion matrix plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    plots_dir: str | Path = "reports",
) -> dict:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }

    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_test, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(metrics["confusion_matrix"]),
        display_labels=["Susceptible", "Resistant"],
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} – Confusion Matrix")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(plots_dir / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.close(fig)

    print(
        f"[{model_name}] Acc={metrics['accuracy']} F1={metrics['f1']} "
        f"ROC-AUC={metrics.get('roc_auc', 'N/A')} MCC={metrics['mcc']}"
    )
    return metrics
