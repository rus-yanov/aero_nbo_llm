# src/evaluation/metrics.py
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    average_precision_score,
)


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> Dict[str, float]:

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # бинарные предсказания по порогу
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {}

    # AUC ROC
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    # LogLoss
    metrics["log_loss"] = float(log_loss(y_true, y_proba))

    # Accuracy при фиксированном пороге
    metrics[f"accuracy@{threshold}"] = float(accuracy_score(y_true, y_pred))

    # PR-AUC (Average Precision)
    metrics["average_precision"] = float(average_precision_score(y_true, y_proba))

    return metrics